import torch
import numpy as np
import pandas as pd
from utils_extra import Dimer_split_seqs, extract_rna_ernie_for_prediction, extract_rna_ernie_single, setup_rna_ernie_model
import paddle
import subprocess
import re


def find_most_available_gpu():

   try:
       cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
       output = subprocess.check_output(cmd.split()).decode('utf-8')
       
       gpus_info = []
       for line in output.strip().split('\n'):
           if line:
               parts = line.split(',')
               if len(parts) == 3:
                   try:
                       index = int(parts[0].strip())
                       used_memory = int(parts[1].strip())
                       total_memory = int(parts[2].strip())
                       gpus_info.append({'index': index, 'used': used_memory, 'total': total_memory})
                   except ValueError:
                       continue
       
       if not gpus_info:
           print("Warning: No GPU device information detected.")
           return -1
           
       gpus_info.sort(key=lambda x: x['used'])
       most_available_gpu_index = gpus_info[0]['index']
       print(f"Most available GPU detected: {most_available_gpu_index} (Used memory: {gpus_info[0]['used']} MiB / Total memory: {gpus_info[0]['total']} MiB)")
       return most_available_gpu_index

   except FileNotFoundError:
       print("Error: 'nvidia-smi' command not found.")
       return -1
   except subprocess.CalledProcessError as e:
       print(f"Error: Failed to execute 'nvidia-smi' command. Error message: {e}")
       return -1
   except Exception as e:
       print(f"Error: Unknown error occurred while finding available GPU: {e}")
       return -1


class ToeholdPredictorSimplified:
   
   def __init__(self, 
                on_model_path,
                off_model_path,
                use_rna_ernie=False,
                vocab_path=None,
                ernie_model_path=None,
                auto_select_gpu=True,
                device_num=None):
       self.device_num = device_num
       self.use_rna_ernie = use_rna_ernie
       
       if auto_select_gpu:
           selected_gpu_idx = find_most_available_gpu()
           if selected_gpu_idx != -1 and torch.cuda.is_available():
               self.device = torch.device(f'cuda:{selected_gpu_idx}')
               torch.cuda.set_device(selected_gpu_idx)
               print(f"✓ Auto-selected GPU: cuda:{selected_gpu_idx}")
           else:
               self.device = torch.device('cpu')
               print("⚠️  Using CPU")
       elif torch.cuda.is_available() and self.device_num is not None:
           if self.device_num < torch.cuda.device_count():
               self.device = torch.device(f'cuda:{self.device_num}')
               torch.cuda.set_device(self.device_num)
               print(f"✓ Manually specified GPU: cuda:{self.device_num}")
           else:
               print(f"Warning: Specified device_num ({self.device_num}) out of range, using CPU")
               self.device = torch.device('cpu')
       else:
           self.device = torch.device('cpu')
           print("CUDA unavailable, using CPU")

       
       print(f"\nLoading PyTorch prediction models to {self.device}...")
       self.on_model = torch.load(on_model_path, map_location=self.device)
       self.on_model.eval()
       print(f"✓ ON model loaded")
       
       self.off_model = torch.load(off_model_path, map_location=self.device)
       self.off_model.eval()
       print(f"✓ OFF model loaded")
       
       self.vocab_path = vocab_path
       self.ernie_model_path = ernie_model_path
       
       self.ernie_model = None
       self.batch_converter = None
       
       if self.device.type == 'cuda':
           paddle.device.set_device(f'gpu:{self.device.index}')
           print(f"✓ PaddlePaddle set to use GPU: {self.device.index}")
       else:
           paddle.device.set_device('cpu')
           print("✓ PaddlePaddle set to use CPU")
           
       if self.use_rna_ernie:
           if not vocab_path or not ernie_model_path:
               raise ValueError("vocab_path and ernie_model_path must be provided when using RNA-Ernie")
           
           print("\nInitializing RNA-Ernie components...")
           self.ernie_model, self.batch_converter = setup_rna_ernie_model(
               vocab_path=vocab_path,
               ernie_model_path=ernie_model_path,
               k_mer=1,
               max_seq_len=115,
               is_pad=True,
               st_pos=0
           )
           print("✓ RNA-Ernie components initialized")
   
   def build_structure_array(self, seq: str) -> np.ndarray:
       """Build RNA structure feature matrix (seq_len x seq_len) - consistent with training"""
       seq = seq.upper().replace('T', 'U')
       N = len(seq)
       struct_array = np.zeros((N, N), dtype=np.float32)
       
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
   
   def extract_features_for_single_sequence(self, mRNA: str):
       """
       Extract all features for a single sequence
       
       Args:
           mRNA: Single mRNA sequence string (115bp)
       
       Returns:
           features: Dimer features [1, 2, 115] (float32)
           ernies: RNA-Ernie embeddings [1, 115, hidden_size] (float32) or None
           structures: Structure matrix [1, 115, 115] (float32)
       """
       if len(mRNA) != 115:
           raise ValueError(f"Sequence length must be 115bp, current length is {len(mRNA)}bp")
       
       feature = Dimer_split_seqs(mRNA)
       feature = np.array(feature, dtype=np.int32)
       features = np.array([feature], dtype=np.float32)
       
       structure = self.build_structure_array(mRNA)
       structures = np.array([structure], dtype=np.float32)
       
       ernies = None
       if self.use_rna_ernie:
           ernie_embedding = extract_rna_ernie_single(
               sequence=mRNA,
               ernie_model=self.ernie_model,
               batch_converter=self.batch_converter
           )
           if ernie_embedding is None:
               raise RuntimeError("RNA-Ernie embedding extraction failed")
           ernies = np.array([ernie_embedding], dtype=np.float32)
       
       return features, ernies, structures
   
   def extract_features_for_batch(self, mRNAs: list):
       """
       Extract all features for batch sequences
       
       Args:
           mRNAs: List of mRNA sequences
       
       Returns:
           features: Dimer features [N, 2, 115] (float32)
           ernies: RNA-Ernie embeddings [N, 115, hidden_size] (float32) or None
           structures: Structure matrices [N, 115, 115] (float32)
           valid_indices: List of original indices for valid sequences
       """
       features_list = []
       structures_list = []
       valid_indices = []
       
       for i, mRNA in enumerate(mRNAs):
           if len(mRNA) != 115:
               print(f'Sequence {i} length is {len(mRNA)}, skipping')
               continue
           
           feature = Dimer_split_seqs(mRNA)
           feature = np.array(feature, dtype=np.int32)
           features_list.append(feature)
           
           structure = self.build_structure_array(mRNA)
           structures_list.append(structure)
           
           valid_indices.append(i)
       
       if len(features_list) == 0:
           raise ValueError("No valid sequences (length must be 115bp)")
       
       features = np.array(features_list, dtype=np.float32)
       structures = np.array(structures_list, dtype=np.float32)
       
       ernies = None
       if self.use_rna_ernie:
           print("\nExtracting RNA-Ernie embeddings in batch...")
           valid_sequences = [mRNAs[i] for i in valid_indices]
           ernies = extract_rna_ernie_for_prediction(
               mRNAs_list=valid_sequences,
               vocab_path=self.vocab_path,
               ernie_model_path=self.ernie_model_path,
               batch_size=64,
               k_mer=1,
               max_seq_len=115
           )
           ernies = ernies.astype(np.float32)
       
       return features, ernies, structures, valid_indices
   
   def construct_full_sequence_from_row(self, row):
       """Construct complete 115bp sequence from CSV row"""
       full_seq = (
           str(row['loop1']) + 
           str(row['switch']) + 
           str(row['loop2']) + 
           str(row['stem1']) + 
           str(row['atg']) + 
           str(row['stem2']) + 
           str(row['linker']) + 
           str(row['post_linker'])
       )
       return full_seq
   
   def predict_single(self, mRNA: str):
       """
       Predict a single mRNA sequence
       
       Args:
           mRNA: Single mRNA sequence string (115bp)
       
       Returns:
           dict: {'ON': float, 'OFF': float, 'ON_OFF_Ratio': float, 'sequence_length': int}
       """
       print(f"\n{'='*70}")
       print(f"Single sequence prediction")
       print(f"{'='*70}\n")
       
       print("Extracting features...")
       features, ernies, structures = self.extract_features_for_single_sequence(mRNA)
       
       print(f"✓ Dimer features shape: {features.shape}")
       print(f"✓ Structure matrix shape: {structures.shape}")
       if ernies is not None:
           print(f"✓ Ernie embeddings shape: {ernies.shape}")
       
       features_tensor = torch.from_numpy(features).to(self.device)
       structures_tensor = torch.from_numpy(structures).to(self.device)
       ernies_tensor = torch.from_numpy(ernies).to(self.device) if ernies is not None else None
       
       print("\nPerforming prediction...")
       with torch.no_grad():
           on_pred = self.on_model(features_tensor, structures_tensor, ernies_tensor)
           off_pred = self.off_model(features_tensor, structures_tensor, ernies_tensor)
           
           on_pred = on_pred.squeeze().cpu().item()
           off_pred = off_pred.squeeze().cpu().item()
       
       on_value = on_pred
       off_value = off_pred
       ratio = on_value / off_value if off_value > 1e-6 else float('inf')
       
       result = {
           'ON': float(on_value),
           'OFF': float(off_value),
           'ON_OFF_Ratio': float(ratio),
           'sequence_length': len(mRNA)
       }
       
       print(f"\n{'='*70}")
       print(f"✓ Prediction complete!")
       print(f"  - ON:  {result['ON']:.4f}")
       print(f"  - OFF: {result['OFF']:.4f}")
       print(f"  - Ratio: {result['ON_OFF_Ratio']:.4f}")
       print(f"{'='*70}\n")
       
       return result
   
   def predict_from_csv(self, csv_path, output_path=None, batch_size=32):
       """
       Batch prediction from CSV file (process in batches to avoid memory overflow)
       
       Args:
           csv_path: Input CSV file path
           output_path: Output CSV file path
           batch_size: Batch size for prediction
       
       Returns:
           pandas DataFrame: DataFrame containing prediction results
       """
       print(f"\n{'='*70}")
       print(f"Starting batch prediction: {csv_path}")
       print(f"Batch size: {batch_size}")
       print(f"{'='*70}\n")
       
       df = pd.read_csv(csv_path)
       print(f"✓ Data loaded: {len(df)} entries")
       
       print("\nConstructing sequences...")
       sequences = []
       for idx, row in df.iterrows():
           try:
               full_seq = self.construct_full_sequence_from_row(row)
               sequences.append(full_seq)
           except Exception as e:
               print(f"❌ Row {idx} sequence construction failed: {e}")
               sequences.append(None)
       
       print("\nExtracting features...")
       features, ernies, structures, valid_indices = self.extract_features_for_batch(sequences)
       
       print(f"✓ Valid sequences: {len(valid_indices)}/{len(df)}")
       print(f"✓ Dimer features shape: {features.shape}")
       print(f"✓ Structure matrices shape: {structures.shape}")
       if ernies is not None:
           print(f"✓ Ernie embeddings shape: {ernies.shape}")
       
       print(f"\nPerforming batch prediction (batch_size={batch_size})...")
       num_samples = len(valid_indices)
       num_batches = (num_samples + batch_size - 1) // batch_size
       
       all_on_preds = []
       all_off_preds = []
       
       with torch.no_grad():
           for batch_idx in range(num_batches):
               start_idx = batch_idx * batch_size
               end_idx = min(start_idx + batch_size, num_samples)
               
               batch_features = torch.from_numpy(features[start_idx:end_idx]).to(self.device)
               batch_structures = torch.from_numpy(structures[start_idx:end_idx]).to(self.device)
               batch_ernies = None
               if ernies is not None:
                   batch_ernies = torch.from_numpy(ernies[start_idx:end_idx]).to(self.device)
               
               batch_on_pred = self.on_model(batch_features, batch_structures, batch_ernies)
               batch_off_pred = self.off_model(batch_features, batch_structures, batch_ernies)
               
               all_on_preds.append(batch_on_pred.squeeze().cpu().numpy())
               all_off_preds.append(batch_off_pred.squeeze().cpu().numpy())
               
               if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                   print(f"  Completed batches: {batch_idx + 1}/{num_batches} "
                         f"({end_idx}/{num_samples} sequences)")
               
               del batch_features, batch_structures, batch_ernies
               del batch_on_pred, batch_off_pred
               torch.cuda.empty_cache()
       
       on_preds = np.concatenate(all_on_preds)
       off_preds = np.concatenate(all_off_preds)
       
       print(f"✓ All batch predictions complete")
       
       on_values = on_preds
       off_values = off_preds
       ratios = np.where(off_values > 1e-6, on_values / off_values, np.inf)
       
       df['Predicted_ON'] = None
       df['Predicted_OFF'] = None
       df['Predicted_ON_OFF_Ratio'] = None
       
       for i, idx in enumerate(valid_indices):
           df.at[idx, 'Predicted_ON'] = float(on_values[i])
           df.at[idx, 'Predicted_OFF'] = float(off_values[i])
           df.at[idx, 'Predicted_ON_OFF_Ratio'] = float(ratios[i])
       
       if output_path is None:
           output_path = csv_path.replace('.csv', '_predictions.csv')
       
       df.to_csv(output_path, index=False)
       
       print(f"\n{'='*70}")
       print(f"✓ Prediction complete!")
       print(f"  - Valid predictions: {len(valid_indices)}/{len(df)}")
       print(f"  - Results saved: {output_path}")
       print(f"{'='*70}\n")
       
       valid_df = df.loc[valid_indices]
       print("Prediction statistics:")
       print(valid_df[['Predicted_ON', 'Predicted_OFF', 'Predicted_ON_OFF_Ratio']].describe())
       
       return df


if __name__ == '__main__':
   
   
   USE_RNA_ERNIE = True
   ERNIE_VOCAB_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt"
   ERNIE_MODEL_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final"
   
   predictor = ToeholdPredictorSimplified(
       on_model_path='/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/model/adjusted_on_pearson_mse_structure_9_pcc=0.8279.pth',
       off_model_path='/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/model/off_pearson_mse_ernie_structure_12_pcc=0.7123.pth',

       
       use_rna_ernie=USE_RNA_ERNIE,
       vocab_path=ERNIE_VOCAB_PATH if USE_RNA_ERNIE else None,
       ernie_model_path=ERNIE_MODEL_PATH if USE_RNA_ERNIE else None,
       
       auto_select_gpu=True
   )
   
   test_mRNA = "ACGTACGT" * 14 + "ACG"
   result = predictor.predict_single(test_mRNA)
   print(f"Prediction result: {result}")
   
   csv_path = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/data/Toehold_mRNA_Dataset_clean.csv'
   results_df = predictor.predict_from_csv(
       csv_path=csv_path,
       output_path='predictions_simplified.csv',
       batch_size=32
   )

def get_rna_prediction(sequence: str, predictor_instance):
   """
   Input RNA sequence, return ON/OFF values and ratio
   """
   try:
       if len(sequence) != 115:
           return {"error": f"Sequence length must be 115bp, current length is {len(sequence)}"}
       
       prediction = predictor_instance.predict_single(sequence)
       
       return {
           "on_value": prediction['ON'],
           "off_value": prediction['OFF'],
           "ratio": prediction['ON_OFF_Ratio']
       }
   except Exception as e:
       return {"error": str(e)}
