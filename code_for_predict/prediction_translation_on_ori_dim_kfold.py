import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader,Dataset

import numpy as np
import pdb
import pickle
import os
import RNA
from sklearn.model_selection import KFold

def dotbracket_to_matrix(sequence: str, structure: str, use_hbond_strength=False):
    N = len(sequence)
    matrix = np.zeros((N, N), dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            if use_hbond_strength:
                pair = (sequence[j], sequence[i])
                if pair in [('A', 'U'), ('U', 'A'), ('G', 'U'), ('U', 'G')]:
                    val = 2
                elif pair in [('G', 'C'), ('C', 'G')]:
                    val = 3
                else:
                    val = 1
            else:
                val = 1
            matrix[i, j] = matrix[j, i] = val

    return matrix

def predict_structure_matrix(sequence: str, use_hbond_strength=True):
    sequence = sequence.upper().replace('T', 'U')
    fc = RNA.fold_compound(sequence)
    structure, mfe = fc.mfe()
    matrix = dotbracket_to_matrix(sequence, structure, use_hbond_strength)
    
    return matrix

def make_dataset_sequences_bio(mRNAs, ons, offs, on_offs, out_label='on', structure=False):

    features_array = []
    labels_array = []
    structures = []

    max_on = max(ons)
    max_off = max(offs)
    max_on_off = max(on_offs)

    min_off = min(offs)
    min_on = min(ons)
    min_on_off = min(on_offs)

    number = 0

    for mRNA, on, off, on_off in zip(mRNAs, ons, offs, on_offs):

        if len(mRNA) != 115:

            print('length = ', len(mRNA))
            print('sequence = ',mRNA)
            pdb.set_trace()

            continue
        
        feature = Dimer_split_seqs(mRNA)
        feature = np.array(feature)
        feature = feature.astype(int)
        matrix = predict_structure_matrix(sequence=mRNA)
        
        structures.append(matrix)
        
        features_array.append(feature)

        label_on = (on - min_on)/(max_on -  min_on)
        label_off  = (off - min_off)/(max_off -  min_off)
        label_on_off = (on_off - min_on_off)/(max_on_off -  min_on_off)

        if out_label == 'on':
            
            labels_array.append(label_on)

        elif out_label == 'off':
            labels_array.append(label_off)

        elif out_label == 'on_off':
            labels_array.append(label_on_off)

        else:
            print('flag is error')

        number += 1
    
    print('number = ',number)
    
    if structure:
        return np.array(features_array), np.array(labels_array), np.array(structures)
    
    else:
        return np.array(features_array), np.array(labels_array)


def read_data(filename, qc=False):

    import math

    mRNAs = []
    ons = []

    offs = []
    on_offs = []

    df = pd.read_csv(filename)

    number = 0
    if not qc:
        for loop1,switch,loop2,stem1,atg,stem2,linker,post_linker,on,off,on_off in zip(df['loop1'], df['switch'], df['loop2'], df['stem1'], df['atg'], df['stem2'], df['linker'], df['post_linker'], df['ON'], df['OFF'], df['ON_OFF']):
            
            on = float(on)
            off = float(off)
            on_off = float(on_off)

            if math.isnan(on) or math.isnan(off) or math.isnan(on_off):
                
                print(f'on is {on}!!!\noff is {off}!!!\non_off is {on_off}!!!')
                
                continue

            mRNAs.append(loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker)
            ons.append(on)
            offs.append(off)
            on_offs.append(on_off)

            number += 1

        print('number is ', number)
        
    else:
        for loop1,switch,loop2,stem1,atg,stem2,linker,post_linker,on,off,on_off in zip(df['loop1'], df['switch'], df['loop2'], df['stem1'], df['atg'], df['stem2'], df['linker'], df['post_linker'], df['QC_ON'], df['QC_OFF'], df['QC_ON_OFF']):
            
            on = float(on)
            off = float(off)
            on_off = float(on_off)

            if math.isnan(on) or math.isnan(off) or math.isnan(on_off):
                
                print(f'on is {on}!!!\noff is {off}!!!\non_off is {on_off}!!!')
                
                continue

            mRNAs.append(loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker)
            ons.append(on)
            offs.append(off)
            on_offs.append(on_off)

            number += 1

        print('number is ', number)
        
    return mRNAs, ons, offs, on_offs

class CustomDataset(Dataset):
    def __init__(self, features, labels):

        self.features = features
        self.labels = labels

    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):

        feature = self.features[idx]
        label = self.labels[idx]


        return feature, label


def train(params, features_array, labels_array):

    patience = 50
    
    print('params = ',params)

    test_pearson_kfold = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')
        no_improve_epochs = 0

        print(f"Fold {fold + 1}/{k_folds}")

        print('size of train dataset is: ', len(train_indices))
        print('size of test dataset is: ', len(val_indices))

        train_dataset = CustomDataset(features_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], labels_array[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        print('training set length = ',len(train_loader))
        print('test set length = ',len(test_loader))

        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_translation(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train =[]
        loss_test = []

        metric = []

        for epoch in range(params['train_epochs_num']):
            
            
            if epoch > 0 and epoch % 100 == 0:

                for param_group in opt_gen.param_groups:

                    print('adjusting learning rate')
                    param_group['lr'] = param_group['lr'] / 2.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0

            loss_mse = 0
            loss_pier = 0
            
            gen.train()

            for data, target in train_loader:
                
                data = data.to(device)
                target = target.to(device)

                output = gen(data)
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(),output=output.float())

                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()

                if loss_kind == 'pearson':
                    loss_all = -loss_pi

                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen
                
                elif loss_kind == 'mse':
                    loss_all = loss_gen

                else:
                    print('incorrect loss function type, please check!')

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
                
            loss_train.append(loss_train_one_epoch/len(train_loader))

            if epoch % 10 == 0:
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                        )
            
            gen.eval()

            targets = []
            outputs = []

            for data, target in test_loader:
                
                data = data.to(device)
                target = target.to(device)

                output = gen(data)
                output = torch.squeeze(output, dim=1)
                loss_gen = loss_fc(target, output)
                

                targets.append(target.detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())

                loss_test_one_epoch += loss_gen.item()
            
            correlation_coefficient = compute_correlation_coefficient(np.concatenate(targets, axis=0), np.concatenate(outputs, axis=0) )

            loss_test.append(loss_test_one_epoch/len(test_loader))

            if loss_test_one_epoch/len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch/len(test_loader)

                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:
                
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                        )
            
            metric.append(correlation_coefficient)

            global pcc
            if correlation_coefficient > pcc:

                pcc = correlation_coefficient
                
                if loss_kind == 'pearson':
                    torch.save(gen,'../model/on_pearson_predict_{0}_metric={1:.6f}.pth'.format(epoch, correlation_coefficient))
                
                elif loss_kind == 'pearson_mse':
                    torch.save(gen,'../model/on_pearson_mse_predict_{0}_metric={1:.6f}.pth'.format(epoch, correlation_coefficient))
                
                elif loss_kind == 'mse':
                    torch.save(gen,'../model/on_mse_predict_{0}_metric={1:.6f}.pth'.format(epoch, correlation_coefficient))
                
                else:
                    print('loss function type error, please check!')

            if no_improve_epochs > 0 and no_improve_epochs % 10:

                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr']*0.85


            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


        dict2 = {'correlation_coefficient':max(metric),'min_train_loss':min(loss_train),'min_test_loss':min(loss_test),'k_fold':fold+1}
        
        if loss_kind == 'pearson':
            write_good_record(dict1=params,dict2=dict2,file_path='../result/on_good_record_metric_pearson_ori_dim.txt')
        
        elif loss_kind == 'pearson_mse':
            write_good_record(dict1=params,dict2=dict2,file_path='../result/on_good_record_metric_pearson_mse_ori_dim.txt')
        
        elif loss_kind == 'mse':
            write_good_record(dict1=params,dict2=dict2,file_path='../result/on_good_record_metric_mse_ori_dim.txt')
        
        else:
            print('loss function type error, please check!')
        
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)



pcc = 0.7

if __name__ == '__main__':
 
    filename = '/home/liangce/lx/Promoter_mRNA_synthetic/data/Toehold_mRNA_Dataset_clean.csv'
    mRNAs, ons, offs, on_offs = read_data(filename=filename)
    features_array, labels_array = make_dataset_sequences_bio(mRNAs, ons, offs, on_offs)

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    params = {'device_num': 1, 'dropout_rate1': 0.47546683788292143, 'dropout_rate2': 0.18657287855258853, 'dropout_rate_fc': 0.4754511996566958, 'embedding_dim1': 128, 'embedding_dim2': 128, 'fc_hidden1': 144, 'fc_hidden2': 8, 'hidden_dim1': 64, 'hidden_dim2': 512, 'l2_regularization': 5e-06, 'latent_dim1': 256, 'latent_dim2': 512, 'num_head1': 8, 'num_head2': 16, 'seq_len': 115, 'train_base_learning_rate': 0.0013977788762949133, 'train_batch_size': 512, 'train_epochs_num': 500, 'transformer_num_layers1': 3, 'transformer_num_layers2': 11}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =',device)

    loss_kind = 'pearson_mse'

    train(params, features_array=features_array, labels_array=labels_array)

    space = {

        'train_batch_size':hp.choice('train_batch_size',[512]),
        'seq_len':hp.choice('seq_len',[115]),  
        'device_num':hp.choice('device_num',[1]),
        'train_epochs_num':hp.choice('train_epochs_num',[500]),

        'train_base_learning_rate': hp.loguniform('train_base_learning_rate', -7, -4),

        'dropout_rate1': hp.uniform('dropout_rate1', 0.1, 0.5),
        'dropout_rate2': hp.uniform('dropout_rate2', 0.1, 0.5),
        'dropout_rate_fc': hp.uniform('dropout_rate_fc', 0.1, 0.5),

        'transformer_num_layers1': hp.randint('transformer_num_layers1',1, 12),
        'transformer_num_layers2': hp.randint('transformer_num_layers2',1, 12),
        
        'l2_regularization': hp.choice('l2_regularization', [5e-5,2e-5,5e-6]),

        'num_head1': hp.choice('num_head1', [2, 4, 8, 16]),
        'num_head2': hp.choice('num_head2', [2, 4, 8, 16]),

        'hidden_dim1': hp.choice('hidden_dim1',[64,128,256,512,1024]),
        'latent_dim1': hp.choice('latent_dim1', [64,128, 256,512]),
        'embedding_dim1': hp.choice('embedding_dim1',[64,128, 256,512]),

        'hidden_dim2': hp.choice('hidden_dim2',[128,256,512,1024]),
        'latent_dim2': hp.choice('latent_dim2', [64, 128, 256,512]),
        'embedding_dim2': hp.choice('embedding_dim2',[64, 128, 256,512]),

        'fc_hidden1': hp.randint('fc_hidden1',64, 256),
        'fc_hidden2': hp.randint('fc_hidden2',8, 64)
    }

    trials = Trials()

    objective = lambda params: train(params, features_array=features_array, labels_array=labels_array)
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    print('best parameters:', best)
