import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import KFold
import random

RANDOM_SEED = 42

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to: {seed}")

USE_RNA_ERNIE = True
VOCAB_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt"
ERNIE_MODEL_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final"
EMBEDDING_SAVE_DIR = "ernie_embeddings_batches"
TARGET_GPUS_FOR_ERNIE = [1, 2, 6]

def build_structure_array(seq: str) -> np.ndarray:
    seq = seq.upper().replace('T', 'U')
    N = len(seq)
    struct_array = np.zeros((N, N), dtype=int)
    
    pair_map = {
        ('A', 'U'): 2, ('U', 'A'): 2,
        ('G', 'C'): 3, ('C', 'G'): 3,
        ('G', 'U'): 2, ('U', 'G'): 2
    }
    
    for i in range(N):
        for j in range(N):
            pair = (seq[i], seq[j])
            if pair in pair_map:
                struct_array[i, j] = pair_map[pair]
    
    return struct_array

def make_dataset_sequences_bio(mRNAs, ons, offs, on_offs, rna_ernie_embeddings=None, out_label='off', structure=False):
    features_array = []
    labels_array = []
    structures = []
    ernies = []
    
    max_on = max(ons)
    max_off = max(offs)
    max_on_off = max(on_offs)
    min_off = min(offs)
    min_on = min(ons)
    min_on_off = min(on_offs)
    
    number = 0
    
    for i, (mRNA, on, off, on_off) in enumerate(zip(mRNAs, ons, offs, on_offs)):
        if len(mRNA) != 115:
            print(f'Warning: Sequence {i} length is not 115, skipping')
            continue
        
        feature = Dimer_split_seqs(mRNA)
        feature = np.array(feature, dtype=np.int32)
        features_array.append(feature)
        
        if rna_ernie_embeddings is not None:
            ernie_embedding = rna_ernie_embeddings[i]
            ernies.append(ernie_embedding)
        
        if structure:
            matrix = build_structure_array(seq=mRNA)
            structures.append(matrix)
        
        label_on = (on - min_on) / (max_on - min_on)
        label_off = (off - min_off) / (max_off - min_off)
        label_on_off = (on_off - min_on_off) / (max_on_off - min_on_off)
        
        if out_label == 'on':
            labels_array.append(label_on)
        elif out_label == 'off':
            labels_array.append(label_off)
        elif out_label == 'on_off':
            labels_array.append(label_on_off)
        else:
            raise ValueError('out_label must be "on", "off", or "on_off"')
        
        number += 1
    
    print('number = ', number)
    
    features_array = np.array(features_array, dtype=np.float32)
    labels_array = np.array(labels_array, dtype=np.float32)
    
    if rna_ernie_embeddings is not None:
        ernies = np.array(ernies, dtype=np.float32)
        if structure:
            structures = np.array(structures, dtype=np.float32)
            return features_array, ernies, structures, labels_array
        else:
            return features_array, ernies, labels_array
    else:
        if structure:
            structures = np.array(structures, dtype=np.float32)
            return features_array, structures, labels_array
        else:
            return features_array, labels_array

def read_data(filename):
    import math
    
    mRNAs = []
    ons = []
    offs = []
    on_offs = []
    original_indices = []
    df = pd.read_csv(filename)
    
    number = 0
    
    for idx, (loop1, switch, loop2, stem1, atg, stem2, linker, post_linker, on, off, on_off) in enumerate(
        zip(df['loop1'], df['switch'], df['loop2'], df['stem1'], df['atg'], 
            df['stem2'], df['linker'], df['post_linker'], df['ON'], df['OFF'], df['ON_OFF'])
    ):
        on = float(on)
        off = float(off)
        on_off = float(on_off)
        
        if math.isnan(on) or math.isnan(off) or math.isnan(on_off):
            print(f'Skipping invalid data: on={on}, off={off}, on_off={on_off}')
            continue
        
        mRNAs.append(loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker)
        ons.append(on)
        offs.append(off)
        on_offs.append(on_off)
        original_indices.append(idx)
        number += 1
    
    print('Valid data count:', number)
    return mRNAs, ons, offs, on_offs, original_indices

class CustomDataset(Dataset):
    def __init__(self, features, ernies, structures, labels):
        self.features = features
        self.ernies = ernies
        self.structures = structures
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        ernie = self.ernies[idx] if self.ernies is not None else None
        structure = self.structures[idx]
        label = self.labels[idx]
        
        if ernie is not None:
            return feature, ernie, structure, label
        else:
            return feature, structure, label

def file_detection(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            pass
    print(f"File ensured to exist: {file_path}")

def train(params, features_array, ernie_embeddings, structure_array, labels_array, random_seed=42, model_init_seed=None):
    set_random_seed(random_seed)
    
    patience = 50
    print('params = ', params)
    print(f'Global random seed = {random_seed}')
    
    loss_save_dir = '../result/loss_records'
    os.makedirs(loss_save_dir, exist_ok=True)
    
    embedding_type = 'ernie' if ernie_embeddings is not None else 'ori_dim'
    model_init_str = f'_init{model_init_seed}' if model_init_seed is not None else ''
    csv_filename = os.path.join(
        loss_save_dir,
        f'off_all_folds_loss_record_{loss_kind}_{embedding_type}_seed{random_seed}{model_init_str}.csv'
    )
    
    if not os.path.exists(csv_filename):
        header_df = pd.DataFrame(columns=[
            'fold', 'epoch', 'train_loss', 'train_mse', 'train_pearson',
            'test_loss', 'test_correlation', 'test_mse', 'test_r2', 'test_spearman',
            'fold_seed', 'model_init_seed'
        ])
        header_df.to_csv(csv_filename, mode='w', index=False)
        print(f"CSV file created: {csv_filename}")
    
    test_pearson_kfold = []
    
    result_file = f'../result/off_good_record_metric_{loss_kind}_ori_dim_structure.txt'
    if ernie_embeddings is not None:
        result_file = f'../result/off_good_record_metric_{loss_kind}_ernie_structure.txt'
    
    print(f"Current K-Fold results will be written to: {result_file}")
    print(f"Records for each epoch will be appended in real-time to: {csv_filename}")
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):
        best_val_loss = float('inf')
        no_improve_epochs = 0
        
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        print('Training set size:', len(train_indices))
        print('Test set size:', len(val_indices))
        
        fold_seed = random_seed * 100 + fold
        set_random_seed(fold_seed)
        print(f"🎲 Fold {fold+1} using seed: {fold_seed} (for data splitting)")
        
        if ernie_embeddings is not None:
            train_dataset = CustomDataset(
                features_array[train_indices],
                ernie_embeddings[train_indices],
                structure_array[train_indices],
                labels_array[train_indices]
            )
            test_dataset = CustomDataset(
                features_array[val_indices],
                ernie_embeddings[val_indices],
                structure_array[val_indices],
                labels_array[val_indices]
            )
        else:
            train_dataset = CustomDataset(
                features_array[train_indices],
                None,
                structure_array[train_indices],
                labels_array[train_indices]
            )
            test_dataset = CustomDataset(
                features_array[val_indices],
                None,
                structure_array[val_indices],
                labels_array[val_indices]
            )
        
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)
        
        print('Training set length = ', len(train_loader))
        print('Test set length = ', len(test_loader))
        
        print('start compose simple gan model')
        
        model_seed_for_init = fold_seed if model_init_seed is None else (int(model_init_seed) + fold)
        torch.manual_seed(model_seed_for_init)
        torch.cuda.manual_seed(model_seed_for_init)
        torch.cuda.manual_seed_all(model_seed_for_init)
        print(f"  ⚙️  Model weight initialization seed: {model_seed_for_init} (model_init_seed={model_init_seed})")
        
        gen = predict_transformerv2.Predict_translation_structure(params=params).to(device)
        initialize_weights(gen)
        
        print('successful compose simple gan model')
        
        opt_gen = torch.optim.Adam(
            gen.parameters(), 
            lr=params['train_base_learning_rate'], 
            weight_decay=params['l2_regularization']
        )
        loss_fc = torch.nn.MSELoss()
        
        loss_train = []
        loss_test = []
        metric = []
        all_mse = []
        all_r2 = []
        all_spearman_corr = []
        
        for epoch in range(params['train_epochs_num']):
            if epoch > 0 and epoch % 100 == 0:
                for param_group in opt_gen.param_groups:
                    print('Adjusting learning rate')
                    param_group['lr'] = param_group['lr'] / 2.0
            
            loss_train_one_epoch = 0
            loss_test_one_epoch = 0
            loss_mse = 0
            loss_pier = 0
            
            gen.train()
            
            for batch_data in train_loader:
                if ernie_embeddings is not None:
                    data, ernie, struc, target = batch_data
                    ernie = ernie.to(device)
                else:
                    data, struc, target = batch_data
                
                data = data.to(device)
                target = target.to(device)
                struc = struc.to(device)
                
                output = gen(data, struc, ernie if ernie_embeddings is not None else None)
                output = torch.squeeze(output, dim=1)
                
                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(), output=output.float())
                
                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()
                
                if loss_kind == 'pearson':
                    loss_all = -loss_pi
                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen
                elif loss_kind == 'mse':
                    loss_all = loss_gen
                else:
                    print('Incorrect loss function type, please check!!!')
                
                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()
                
                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
            
            avg_train_loss = loss_train_one_epoch / len(train_loader)
            avg_train_mse = loss_mse / len(train_loader)
            avg_train_pearson = loss_pier / len(train_loader)
            loss_train.append(avg_train_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {avg_train_loss:.6f}****MSE loss: {avg_train_mse:.6f}****Pierxun loss: {avg_train_pearson:.6f}")
            
            gen.eval()
            targets = []
            outputs = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    if ernie_embeddings is not None:
                        data, ernie, struc, target = batch_data
                        ernie = ernie.to(device)
                    else:
                        data, struc, target = batch_data
                    
                    data = data.to(device)
                    target = target.to(device)
                    struc = struc.to(device)
                    
                    output = gen(data, struc, ernie if ernie_embeddings is not None else None)
                    output = torch.squeeze(output, dim=1)
                    loss_gen = loss_fc(target, output)
                    
                    targets.append(target.detach().cpu().numpy())
                    outputs.append(output.detach().cpu().numpy())
                    
                    loss_test_one_epoch += loss_gen.item()
            
            avg_test_loss = loss_test_one_epoch / len(test_loader)
            correlation_coefficient = compute_correlation_coefficient(
                np.concatenate(targets, axis=0), 
                np.concatenate(outputs, axis=0)
            )
            mse, r2, spearman_corr = evaluate_regression_metrics(
                np.concatenate(targets, axis=0), 
                np.concatenate(outputs, axis=0)
            )
            
            loss_test.append(avg_test_loss)
            
            epoch_record = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_mse': avg_train_mse,
                'train_pearson': avg_train_pearson,
                'test_loss': avg_test_loss,
                'test_correlation': correlation_coefficient,
                'test_mse': mse,
                'test_r2': r2,
                'test_spearman': spearman_corr,
                'fold_seed': fold_seed,
                'model_init_seed': model_init_seed if model_init_seed is not None else model_seed_for_init
            }
            
            epoch_df = pd.DataFrame([epoch_record])
            epoch_df.to_csv(csv_filename, mode='a', header=False, index=False)
            
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if epoch % 10 == 0:
                print(f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {avg_test_loss:.6f}********test correlation_coefficient:{correlation_coefficient:.6f}")
            
            metric.append(correlation_coefficient)
            all_mse.append(mse)
            all_r2.append(r2)
            all_spearman_corr.append(spearman_corr)
            
            global pcc
            if correlation_coefficient > pcc:
                pcc = correlation_coefficient
                print(f"New best PCC {pcc:.4f} at fold {fold+1} epoch {epoch+1} (model not saved)")
            
            if no_improve_epochs > 0 and no_improve_epochs % 10 == 0:
                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
            
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        dict2 = {
            'correlation_coefficient': max(metric),
            'mse': min(all_mse),
            'r2': max(all_r2),
            'spearman_corr': max(all_spearman_corr),
            'min_train_loss': min(loss_train),
            'min_test_loss': min(loss_test),
            'k_fold': fold + 1,
            'fold_seed': fold_seed
        }
        
        write_good_record(dict1=params, dict2=dict2, file_path=result_file)
        test_pearson_kfold.append(max(metric))
    
    print(f"\n{'='*60}")
    print(f"All Fold epoch records have been appended in real-time to:")
    print(f"{csv_filename}")
    print(f"{'='*60}")
    
    return -max(test_pearson_kfold)

pcc = 0.7

if __name__ == '__main__':
    filename = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/Toehold_mRNA_Dataset_clean.csv'
    
    GLOBAL_SEED = 42
    MODEL_INIT_SEEDS = [1001, 1002, 1003]
    
    env_global = os.environ.get('GLOBAL_SEED') or os.environ.get('SEED_LIST')
    if env_global:
        GLOBAL_SEED = int(str(env_global).split(',')[0].strip())
    env_init = os.environ.get('MODEL_INIT_SEEDS')
    if env_init:
        MODEL_INIT_SEEDS = [int(x.strip()) for x in env_init.split(',') if x.strip()]
    
    mRNAs, ons, offs, on_offs, original_indices = read_data(filename=filename)
    
    ernie_embeddings = None
    ernie_hidden_size = None
    
    if USE_RNA_ERNIE:
        print("\n========== Starting RNA-Ernie embedding extraction ==========")
        
        physical_gpu = find_and_set_gpu_for_paddle(
            target_gpus=TARGET_GPUS_FOR_ERNIE,
            min_free_mem_gb=50,
            max_gpu_util=5
        )
        
        saved_files, ernie_hidden_size, max_seq_len = extract_rna_ernie_embeddings(
            mRNAs_list=mRNAs,
            valid_original_indices=original_indices,
            vocab_path=VOCAB_PATH,
            ernie_model_path=ERNIE_MODEL_PATH,
            save_dir=EMBEDDING_SAVE_DIR,
            batch_size=64,
            save_batch_threshold=1000
        )
        
        ernie_embeddings = load_rna_ernie_embeddings(
            saved_embedding_files=saved_files,
            valid_original_indices=original_indices,
            save_dir=EMBEDDING_SAVE_DIR
        )
        
        print(f"RNA-Ernie embedding extraction complete. Shape: {ernie_embeddings.shape}")
    
    if USE_RNA_ERNIE and ernie_embeddings is not None:
        features_array, ernie_array, structure_array, labels_array = make_dataset_sequences_bio(
            mRNAs, ons, offs, on_offs,
            rna_ernie_embeddings=ernie_embeddings,
            out_label='off',
            structure=True
        )
    else:
        features_array, structure_array, labels_array = make_dataset_sequences_bio(
            mRNAs, ons, offs, on_offs,
            rna_ernie_embeddings=None,
            out_label='off',
            structure=True
        )
        ernie_array = None
    
    k_folds = 5
    
    params = {
        'device_num': 3,
        'dropout_rate1': 0.485321570018299,
        'dropout_rate2': 0.20608178110433223,
        'dropout_rate_fc': 0.27523219441190655,
        'embedding_dim1': ernie_hidden_size if USE_RNA_ERNIE else 128,
        'embedding_dim2': 128,
        'fc_hidden1': 126,
        'fc_hidden2': 39,
        'hidden_dim1': 256,
        'hidden_dim2': 512,
        'l2_regularization': 5e-05,
        'latent_dim': 64,
        'num_head1': 8,
        'num_head2': 8,
        'seq_len': 115,
        'train_base_learning_rate': 0.0013285018307703882,
        'train_batch_size': 512,
        'train_epochs_num': 500,
        'transformer_num_layers1': 3,
        'transformer_num_layers2': 4
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =', device)
    
    loss_kind = 'pearson_mse'
    
    all_seeds_results = []
    
    print("\n" + "="*80)
    print(f"🎲 Using global random seed = {GLOBAL_SEED} for all data splitting and training (only changing model initialization seed)".center(80))
    print("="*80)

    for init_seed in MODEL_INIT_SEEDS:
        print(f"\n--- Running with model initialization seed {init_seed} ---")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
        
        best_pcc = -train(
            params,
            features_array=features_array,
            ernie_embeddings=ernie_array,
            structure_array=structure_array,
            labels_array=labels_array,
            random_seed=GLOBAL_SEED,
            model_init_seed=init_seed
        )

        all_seeds_results.append({
            'global_seed': GLOBAL_SEED,
            'model_init_seed': init_seed,
            'best_pcc': best_pcc
        })

        print(f"\n✓ Global seed {GLOBAL_SEED} + initialization seed {init_seed} complete, best PCC: {best_pcc:.6f}")
    
    print("\n" + "="*80)
    print("📊 Summary of all random seed experiment results".center(80))
    print("="*80)
    
    results_df = pd.DataFrame(all_seeds_results)
    print(results_df.to_string(index=False))
    print(f"\nAverage PCC: {results_df['best_pcc'].mean():.6f}")
    print(f"Standard deviation: {results_df['best_pcc'].std():.6f}")
    print(f"Maximum value: {results_df['best_pcc'].max():.6f}")
    print(f"Minimum value: {results_df['best_pcc'].min():.6f}")
    
    summary_file = '../result/off_multi_seed_summary.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"\nSummary results saved to: {summary_file}")
    
    print("\nMerging loss records from all seeds...")
    
    loss_save_dir = '../result/loss_records'
    embedding_type = 'ernie' if USE_RNA_ERNIE else 'ori_dim'
    
    all_loss_dfs = []
    pattern = os.path.join(loss_save_dir, f'off_all_folds_loss_record_{loss_kind}_{embedding_type}_seed*.csv')
    matched_files = sorted(glob.glob(pattern))
    for csv_file in matched_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        basename = os.path.basename(csv_file)
        df['source_file'] = basename
        all_loss_dfs.append(df)
    
    if all_loss_dfs:
        combined_df = pd.concat(all_loss_dfs, ignore_index=True)
        combined_file = os.path.join(
            loss_save_dir,
            f'off_all_seeds_combined_{loss_kind}_{embedding_type}.csv'
        )
        combined_df.to_csv(combined_file, index=False)
        print(f"✓ Loss records from all seeds have been merged and saved to: {combined_file}")
        print(f"  Total records: {len(combined_df)}")
    
    print("="*80)
    print("\n🎉 All experiments complete!")
