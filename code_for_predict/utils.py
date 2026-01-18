import scipy as sp
import pdb
import numpy as np
import pandas as pd
import torch
import sys
import os
from scipy.stats import spearmanr
from datetime import datetime
import subprocess
import time

try:
    import paddle
    from paddlenlp.transformers import ErnieModel
    from paddlenlp.data import Stack
except Exception as e:
    paddle = None
    ErnieModel = None
    Stack = None
    print(f"[utils.py] Note: PaddlePaddle/PaddleNLP not installed ({e}). Install if RNA-Ernie is needed.")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from rna_ernie import BatchConverter
    from tokenizer_nuc import NUCTokenizer
    from dataset_utils import seq2input_ids
except Exception as e:
    print(f"[utils.py] Note: Failed to import RNA-Ernie related modules: {e}")

try:
    from rna_ernie import BatchConverter as _BaseBatchConverter
    class PatchedBatchConverter(_BaseBatchConverter):
        def __init__(self, k_mer=1, vocab_path="./data/vocab/vocab_1MER.txt",
                     batch_size=256, max_seq_len=512, is_pad=True, st_pos=0):
            super().__init__(k_mer=k_mer, vocab_path=vocab_path,
                             batch_size=batch_size, max_seq_len=max_seq_len,
                             is_pad=is_pad, st_pos=st_pos)
            self.k_mer = k_mer
            self.vocab_path = vocab_path
            self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
except Exception as e:
    PatchedBatchConverter = None

def one_hot(sequence):
    bases = ['A','T','G','C']
    one_hot_encoded = np.zeros((len(sequence), len(bases)))
    for i, base in enumerate(sequence):
        one_hot_encoded[i, bases.index(base)] = 1
    return one_hot_encoded


def loss_pierxun(output, target):
    target_mean = torch.mean(target)
    outpu_mean = torch.mean(output)

    target_var = torch.std(target)
    output_var = torch.std(output)

    p = torch.mean((output - outpu_mean) * (target - target_mean))

    if output_var == 0 or target_var == 0:
        if output_var == 0 and target_var == 0:
            return torch.tensor(1.0 if torch.mean(output) == torch.mean(target) else 0.0).to(output.device)
        return torch.tensor(0.0).to(output.device)

    p /= (output_var * target_var)
    return p


def text_build_vocab():
    dic = [a for a in 'ATCG']
    dic += [a + b for a in 'ATCG' for b in 'ATCG']
    dic += [a + '0' for a in 'ATCG']
    return dic


def Dimer_split_seqs(seq):
    t = text_build_vocab()
    seq = seq.upper().replace('U', 'T')

    ori_result = []
    dim_result = []

    lens = len(seq)
    for i in range(lens):
        try:
            ori_result.append(t.index(seq[i]))
        except ValueError:
            print(f"Warning: Invalid base '{seq[i]}' in sequence, using -1")
            ori_result.append(-1)

    seq_with_pad = seq + '0'
    wt = 2
    for i in range(lens):
        dimer = seq_with_pad[i:i + wt]
        try:
            dim_result.append(t.index(dimer))
        except ValueError:
            print(f"Warning: Invalid dimer '{dimer}' in sequence, using -1")
            dim_result.append(-1)

    return [ori_result, dim_result]


def write_good_record(dict1, dict2, file_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"\n======== Record Start (Generated: {current_time}) ========\n")
        f.write("Parameters:\n")
        for key, value in dict1.items():
            f.write(f"  {key}: {value}\n")
        f.write("Metrics:\n")
        for key, value in dict2.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"======== Record End (Generated: {current_time}) ========\n")
    print(f"Record successfully written: {file_path}")


def evaluate_regression_metrics(output, label):
    if len(output) != len(label):
        raise ValueError("Output and label lengths are inconsistent")

    if np.isnan(output).any() or np.isnan(label).any():
        print("Warning: Input data contains NaN values, may affect evaluation results.")

    mse = np.mean((output - label) ** 2)

    ss_res = np.sum((label - output) ** 2)
    ss_tot = np.sum((label - np.mean(label)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    if np.std(label) == 0 or np.std(output) == 0:
        spearman_corr = 0.0
    else:
        spearman_corr, _ = spearmanr(label, output)

    return mse, r2, spearman_corr


def compute_correlation_coefficient(output, label):
    target = output
    prediction = label

    has_nan = np.isnan(prediction).any() or np.isnan(target).any()
    if has_nan:
        print("Warning: Arrays contain NaN values, may cause incorrect Pearson coefficient calculation.")

    if np.std(prediction) == 0:
        return 0.0
    if np.std(target) == 0:
        return 0.0

    pearson_coefficient = np.corrcoef(target, prediction)[0, 1]
    return pearson_coefficient


def get_gpu_info():
    try:
        command = "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(command.split()).decode('utf-8').strip().split('\n')
        gpu_info = []
        for line in output:
            parts = line.split(',')
            if len(parts) == 4:
                idx = int(parts[0].strip())
                mem_used = int(parts[1].strip())
                mem_total = int(parts[2].strip())
                gpu_util = int(parts[3].strip())
                gpu_info.append({'index': idx, 'mem_used': mem_used, 'mem_total': mem_total, 'gpu_util': gpu_util})
        return gpu_info
    except Exception as e:
        print(f"Failed to get GPU info: {e}")
        return []


def find_and_set_gpu_for_paddle(target_gpus=[1, 2], min_free_mem_gb=50, max_gpu_util=5):
    if paddle is None:
        print("PaddlePaddle not installed, skipping GPU setup.")
        return None

    selected_gpu = None
    max_retries = 5
    retries = 0

    while selected_gpu is None and retries < max_retries:
        gpu_infos = get_gpu_info()
        available_gpus_for_task = []

        print(f"\n--- Checking GPU status (Target GPUs: {target_gpus}) ---")
        for info in gpu_infos:
            if info['index'] in target_gpus:
                free_mem_gb = (info['mem_total'] - info['mem_used']) / 1024
                print(f"GPU {info['index']}: Used memory {info['mem_used']}MiB / Total memory {info['mem_total']}MiB (Free {free_mem_gb:.2f}GB), GPU utilization {info['gpu_util']}%")
                if free_mem_gb >= min_free_mem_gb and info['gpu_util'] <= max_gpu_util:
                    available_gpus_for_task.append(info['index'])

        if available_gpus_for_task:
            selected_gpu = min(available_gpus_for_task)
            print(f"Found available GPU: Physical GPU {selected_gpu}.")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
            try:
                paddle.set_device(f"gpu:{selected_gpu}")
            except Exception:
                pass
            return selected_gpu
        else:
            retries += 1
            if retries < max_retries:
                print(f"No suitable GPU found, retry {retries}/{max_retries}, waiting 30 seconds...")
                time.sleep(30)
            else:
                print(f"Maximum retries reached ({max_retries}), unable to find available GPU.")
    return None


def setup_rna_ernie_model(vocab_path: str,
                          ernie_model_path: str,
                          k_mer: int = 1,
                          max_seq_len: int = 115,
                          is_pad: bool = True,
                          st_pos: int = 0):
    if ErnieModel is None or paddle is None:
        raise ImportError("PaddlePaddle/PaddleNLP not installed, cannot use RNA-Ernie.")

    print(f"Loading BatchConverter (k_mer={k_mer}, vocab={vocab_path}, max_seq_len={max_seq_len})")
    converter = BatchConverter(
        k_mer=k_mer,
        vocab_path=vocab_path,
        batch_size=256,
        max_seq_len=max_seq_len,
        is_pad=is_pad,
        st_pos=st_pos
    )

    print(f"Loading RNA-Ernie model: {ernie_model_path}")
    ernie_model = ErnieModel.from_pretrained(ernie_model_path)
    ernie_model.eval()
    print("✓ RNA-Ernie model loaded")
    return ernie_model, converter


def tokenize_sequence(sequence: str, nuc_tokenizer: NUCTokenizer, max_seq_len: int = 115) -> list:
    seq = sequence[:max_seq_len - 2]
    seq = seq.upper().replace("U", "T")
    input_ids = seq2input_ids(seq, nuc_tokenizer)
    return input_ids


def extract_rna_ernie_embeddings(mRNAs_list,
                                 valid_original_indices,
                                 vocab_path,
                                 ernie_model_path,
                                 save_dir="ernie_embeddings_batches",
                                 batch_size=64,
                                 save_batch_threshold=1000,
                                 k_mer=1,
                                 max_seq_len=115):
    if ErnieModel is None or paddle is None:
        raise ImportError("PaddlePaddle/PaddleNLP not installed, cannot extract RNA-Ernie embeddings.")

    print("\n--- Starting RNA-Ernie embedding extraction (batch save) ---")
    if not paddle.is_compiled_with_cuda():
        print("Warning: Not in CUDA environment, speed may be slow.")

    converter = BatchConverter(
        k_mer=k_mer,
        vocab_path=vocab_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        is_pad=True,
        st_pos=0
    )

    try:
        rna_ernie = ErnieModel.from_pretrained(ernie_model_path)
        print("RNAErnie model loaded successfully!")
    except Exception as e:
        print(f"Failed to load RNAErnie model! Error: {e}")
        return [], None, None

    rna_ernie.eval()
    try:
        ernie_hidden_size = rna_ernie.config["hidden_size"]
    except Exception:
        ernie_hidden_size = getattr(rna_ernie.config, "hidden_size", None)

    print(f"RNAErnie model hidden layer size: {ernie_hidden_size}")
    print(f"RNAErnie maximum sequence length: {max_seq_len}")

    data_for_ernie = [(f"RNA_{original_idx}", seq) for original_idx, seq in zip(valid_original_indices, mRNAs_list)]

    os.makedirs(save_dir, exist_ok=True)
    batch_save_counter = 0
    saved_embedding_files = []
    current_batch_data_temp = []

    with paddle.no_grad():
        for batch_idx, (batch_names, _, inputs_ids) in enumerate(converter(data_for_ernie)):

            outputs = rna_ernie(inputs_ids)
            last_hidden_state = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            for i, name in enumerate(batch_names):
                original_idx = int(name.split('_')[1])
                seq_emb = last_hidden_state[i].cpu().numpy()
                current_batch_data_temp.append({
                    'original_index': original_idx,
                    'sequence_embedding': seq_emb
                })

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batches {batch_idx + 1}, temporary accumulated {len(current_batch_data_temp)} entries.")

            if len(current_batch_data_temp) >= save_batch_threshold:
                batch_indices_to_save = np.array([item['original_index'] for item in current_batch_data_temp], dtype=int)
                batch_embeddings_to_save = np.stack([item['sequence_embedding'] for item in current_batch_data_temp])

                file_prefix = f"ernie_batch_{batch_save_counter:04d}"
                np.save(os.path.join(save_dir, f"{file_prefix}_indices.npy"), batch_indices_to_save)
                np.save(os.path.join(save_dir, f"{file_prefix}_embeddings.npy"), batch_embeddings_to_save.astype(np.float32))

                saved_embedding_files.append(file_prefix)
                batch_save_counter += 1
                print(f"Saved batch file: {file_prefix}, contains {len(current_batch_data_temp)} entries.")
                current_batch_data_temp = []

    if current_batch_data_temp:
        batch_indices_to_save = np.array([item['original_index'] for item in current_batch_data_temp], dtype=int)
        batch_embeddings_to_save = np.stack([item['sequence_embedding'] for item in current_batch_data_temp])

        file_prefix = f"ernie_batch_{batch_save_counter:04d}"
        np.save(os.path.join(save_dir, f"{file_prefix}_indices.npy"), batch_indices_to_save)
        np.save(os.path.join(save_dir, f"{file_prefix}_embeddings.npy"), batch_embeddings_to_save.astype(np.float32))

        saved_embedding_files.append(file_prefix)
        print(f"Saved final batch file: {file_prefix}, contains {len(current_batch_data_temp)} entries.")

    print("\n--- All RNA-Ernie embeddings saved ---")
    return saved_embedding_files, ernie_hidden_size, max_seq_len


def load_rna_ernie_embeddings(saved_embedding_files, valid_original_indices, save_dir="ernie_embeddings_batches"):
    print("\n--- Loading and merging RNA-Ernie embeddings ---")
    if not saved_embedding_files:
        print("Warning: No embedding files to load.")
        return None

    sorted_files = sorted(saved_embedding_files, key=lambda x: int(x.split('_')[-1]))
    all_loaded_indices = []
    all_loaded_embeddings_list = []

    for file_prefix in sorted_files:
        indices_path = os.path.join(save_dir, f"{file_prefix}_indices.npy")
        embeddings_path = os.path.join(save_dir, f"{file_prefix}_embeddings.npy")

        if not os.path.exists(indices_path) or not os.path.exists(embeddings_path):
            print(f"Warning: Missing index or embedding file for {file_prefix}, skipping.")
            continue

        loaded_indices = np.load(indices_path)
        loaded_embeddings = np.load(embeddings_path)

        all_loaded_indices.append(loaded_indices)
        all_loaded_embeddings_list.append(loaded_embeddings)

    if not all_loaded_embeddings_list:
        print("Error: Failed to load any embedding content!")
        return None

    combined_loaded_indices = np.concatenate(all_loaded_indices)
    combined_loaded_embeddings = np.concatenate(all_loaded_embeddings_list, axis=0)

    loaded_embedding_map = {idx: emb for idx, emb in zip(combined_loaded_indices, combined_loaded_embeddings)}

    final_embeddings = []
    for idx in valid_original_indices:
        if idx in loaded_embedding_map:
            final_embeddings.append(loaded_embedding_map[idx])
        else:
            print(f"Warning: Embedding for original index {idx} is missing.")
            raise ValueError(f"Embedding for original index {idx} not found, cannot construct final array.")

    final_embeddings = np.array(final_embeddings, dtype=np.float32)
    print(f"Loading complete. Final shape: {final_embeddings.shape}")
    return final_embeddings


def extract_rna_ernie_single(sequence: str,
                                   ernie_model,
                                   batch_converter: BatchConverter):
    if paddle is None:
        raise ImportError("PaddlePaddle not installed, cannot extract RNA-Ernie embeddings.")

    data = [("RNA_SINGLE", sequence)]
    with paddle.no_grad():
        for _, _, input_ids in batch_converter(data):
            outputs = ernie_model(input_ids)
            last_hidden_state = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            emb = last_hidden_state[0].cpu().numpy()
            return emb.astype(np.float32)

    raise RuntimeError("Single sequence embedding extraction failed: batch_converter produced no data.")


def extract_rna_ernie_for_prediction(mRNAs_list,
                                     vocab_path,
                                     ernie_model_path,
                                     batch_size=64,
                                     k_mer=1,
                                     max_seq_len=115):
    if ErnieModel is None or paddle is None:
        raise ImportError("PaddlePaddle/PaddleNLP not installed, cannot extract RNA-Ernie embeddings.")

    print("\n--- Starting RNA-Ernie embedding extraction (prediction mode) ---")
    
    converter = BatchConverter(
        k_mer=k_mer,
        vocab_path=vocab_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        is_pad=True,
        st_pos=0
    )

    try:
        rna_ernie = ErnieModel.from_pretrained(ernie_model_path)
        print("✓ RNA-Ernie model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load RNA-Ernie model: {e}")
        raise

    rna_ernie.eval()
    
    data_for_ernie = [(f"RNA_{i}", seq) for i, seq in enumerate(mRNAs_list)]
    
    all_embeddings = []
    
    with paddle.no_grad():
        for batch_idx, (batch_names, _, inputs_ids) in enumerate(converter(data_for_ernie)):
            outputs = rna_ernie(inputs_ids)
            last_hidden_state = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            batch_embeddings = last_hidden_state.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} / {len(mRNAs_list)} sequences")
    
    final_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"✓ Embedding extraction complete, shape: {final_embeddings.shape}")
    
    return final_embeddings
