import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import pdb
import os
from sklearn.model_selection import KFold

USE_RNA_ERNIE = True
VOCAB_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt"
ERNIE_MODEL_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final"
EMBEDDING_SAVE_DIR = "ernie_embeddings_batches"
TARGET_GPUS_FOR_ERNIE = [1, 2, 6]

CSV_LOG_PATH = '../result/adjust_on2_training_log.csv'

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


def make_dataset_sequences_bio(mRNAs, ons, offs, on_offs, rna_ernie_embeddings=None, out_label='on', structure=False):
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
            print('length = ', len(mRNA))
            print('sequence = ', mRNA)
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
        
        if max_on != min_on:
            label_on = (on - min_on) / (max_on - min_on)
        else:
            label_on = 0.5
            
        if max_off != min_off:
            label_off = (off - min_off) / (max_off - min_off)
        else:
            label_off = 0.5
            
        if max_on_off != min_on_off:
            label_on_off = (on_off - min_on_off) / (max_on_off - min_on_off)
        else:
            label_on_off = 0.5

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


def initialize_csv_log(csv_path):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        headers = [
            'fold', 'epoch', 
            'train_loss', 'train_mse', 'train_pearson',
            'test_loss', 'test_correlation', 'test_mse', 'test_r2', 'test_spearman',
            'learning_rate'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_path, index=False)
        print(f"CSV log file created: {csv_path}")


def log_to_csv(csv_path, log_data):
    df = pd.DataFrame([log_data])
    df.to_csv(csv_path, mode='a', header=False, index=False)


def train(params, features_array, ernie_embeddings, structure_array, labels_array):
    patience = 50
    
    print('params = ', params)

    test_pearson_kfold = []
    
    initialize_csv_log(CSV_LOG_PATH)
    
    file_detection(file_path='../result/on_good_record_metric_pearson_ori_dim_structure.txt')
    file_detection(file_path='../result/on_good_record_metric_pearson_mse_ori_dim_structure.txt')
    file_detection(file_path='../result/on_good_record_metric_mse_ori_dim_structure.txt')

    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')
        no_improve_epochs = 0

        print(f"Fold {fold + 1}/{k_folds}")
        print('Training set size:', len(train_indices))
        print('Test set size:', len(val_indices))

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
        gen = predict_transformerv2.Predict_translation_structure(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
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
                    ernie = None
                    
                data = data.to(device)
                target = target.to(device)
                struc = struc.to(device)

                output = gen(data, struc, ernie)
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
                    print('Incorrect loss function type, please check!')

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
                
            loss_train.append(loss_train_one_epoch / len(train_loader))

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                )
            
            gen.eval()

            targets = []
            outputs = []

            for batch_data in test_loader:
                if ernie_embeddings is not None:
                    data, ernie, struc, target = batch_data
                    ernie = ernie.to(device)
                else:
                    data, struc, target = batch_data
                    ernie = None
                    
                data = data.to(device)
                target = target.to(device)
                struc = struc.to(device)

                output = gen(data, struc, ernie)
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target, output)
                loss_pi = loss_pierxun(target=target.float(), output=output.float())

                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()

                if loss_kind == 'pearson':
                    loss_all = -loss_pi
                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen
                elif loss_kind == 'mse':
                    loss_all = loss_gen

                targets.append(target.detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())

                loss_test_one_epoch += loss_all.item()

            correlation_coefficient = compute_correlation_coefficient(np.concatenate(targets, axis=0), np.concatenate(outputs, axis=0))
            mse, r2, spearman_corr = evaluate_regression_metrics(np.concatenate(targets, axis=0), np.concatenate(outputs, axis=0))

            loss_test.append(loss_test_one_epoch / len(test_loader))

            if loss_test_one_epoch / len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch / len(test_loader)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                )
            
            metric.append(correlation_coefficient)
            all_mse.append(mse)
            all_r2.append(r2)
            all_spearman_corr.append(spearman_corr)

            current_lr = opt_gen.param_groups[0]['lr']
            log_data = {
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': loss_train_one_epoch / len(train_loader),
                'train_mse': loss_mse / len(train_loader),
                'train_pearson': loss_pier / len(train_loader),
                'test_loss': loss_test_one_epoch / len(test_loader),
                'test_correlation': correlation_coefficient,
                'test_mse': mse,
                'test_r2': r2,
                'test_spearman': spearman_corr,
                'learning_rate': current_lr
            }
            log_to_csv(CSV_LOG_PATH, log_data)

            global pcc
            if correlation_coefficient > pcc:
                pcc = correlation_coefficient
                
                if loss_kind == 'pearson':
                    torch.save(gen, '../model/on_pearson_structure_{0}_pcc={1:.4f}.pth'.format(epoch, correlation_coefficient))
                elif loss_kind == 'pearson_mse':
                    torch.save(gen, '../model/adjusted_on_pearson_mse_structure_{0}_pcc={1:.4f}.pth'.format(epoch, correlation_coefficient))
                elif loss_kind == 'mse':
                    torch.save(gen, '../model/on_mse_structure_{0}_pcc={1:.4f}.pth'.format(epoch, correlation_coefficient))
                else:
                    print('Loss function type error, please check!')

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
            'k_fold': fold + 1
        }
        
        if loss_kind == 'pearson':
            write_good_record(dict1=params, dict2=dict2, file_path='../result/on_good_record_metric_pearson_ori_dim_structure.txt')
        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params, dict2=dict2, file_path='../result/on_good_record_metric_pearson_mse_ori_dim_structure.txt')
        elif loss_kind == 'mse':
            write_good_record(dict1=params, dict2=dict2, file_path='../result/on_good_record_metric_mse_ori_dim_structure.txt')
        else:
            print('Loss function type error, please check!')
        
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


pcc = 0.81

if __name__ == '__main__':
 
    filename = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/data/Toehold_mRNA_Dataset_clean.csv'
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
            out_label='on',
            structure=True
        )
    else:
        features_array, structure_array, labels_array = make_dataset_sequences_bio(
            mRNAs, ons, offs, on_offs,
            rna_ernie_embeddings=None,
            out_label='on',
            structure=True
        )
        ernie_array = None
    
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    params = {
        'device_num': 1,
        'dropout_rate1': 0.5108978107723199,
        'dropout_rate2': 0.22152898776074903,
        'dropout_rate_fc': 0.2569537131289666,
        'embedding_dim1': ernie_hidden_size if USE_RNA_ERNIE else 768,
        'embedding_dim2': 128,
        'fc_hidden1': 114,
        'fc_hidden2': 39,
        'hidden_dim1': 256,
        'hidden_dim2': 512,
        'l2_regularization': 5e-05,
        'latent_dim': 64,
        'num_head1': 8,
        'num_head2': 8,
        'seq_len': 115,
        'train_base_learning_rate': 0.001719286487497399,
        'train_batch_size': 512,
        'train_epochs_num': 500,
        'transformer_num_layers1': 3,
        'transformer_num_layers2': 4
    }
    
    if USE_RNA_ERNIE and ernie_hidden_size is not None:
        params['embedding_dim1'] = ernie_hidden_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =', device)

    loss_kind = 'pearson_mse'

    print("\n========== Starting training (fixed parameters) ==========")
    final_score = train(params, features_array=features_array, ernie_embeddings=ernie_array, structure_array=structure_array, labels_array=labels_array)
    print(f"\n========== Training complete, final score: {-final_score:.4f} ==========")
