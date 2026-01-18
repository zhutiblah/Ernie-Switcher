import os
import time
from datetime import datetime
import argparse
import torch
from utils import *
from utils_extra import *
import script_utils
import os
import cma
import RNA
from rna_switch_energy import analyze_rna_switch
from Bio import SeqIO
from Bio.Seq import Seq
from switch_predict import ToeholdPredictorSimplified


MFE_ON_RANGE  = (-76.4, -56.9)
MFE_OFF_RANGE = (-39.9, -28.5)
GC_ON_RANGE   = (0.384, 0.486)

ITEM_PEN_MAX  = 0.1
TOTAL_PEN_MAX = 0.3 - 1e-6

def gc_content(seq: str) -> float:
    s = str(seq).upper()
    if len(s) == 0:
        return float("nan")
    return (s.count("G") + s.count("C")) / len(s)

def bounded_violation(x: np.ndarray, lo: float, hi: float, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)

    width = max(hi - lo, eps)
    dist = np.where(x < lo, lo - x, np.where(x > hi, x - hi, 0.0))

    v = dist / (dist + width)
    v = np.where(bad, 1.0, v)
    return v


def dna_reverse_complement_to_rna(dna_seq: str) -> str:
    dna_seq = dna_seq.upper()
    rc_seq = str(Seq(dna_seq).reverse_complement())
    rna_seq = rc_seq.replace("T", "U")
    return rna_seq

def construct_toehold_trigger_from_list(merged_list):
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"
    pre_sequence = "CTCTGGGCTAACTGTCGCGC"
    promoter = "TAATACGACTCACTATAGGG"

    full_sequences = []
    On_sequences = []
    Off_sequences = []
    Ttigger_rna = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"Sequence length should be 45 nt, but found length {len(merged_seq)}: {merged_seq}")
        
        switch = merged_seq[:30]
        stem1 = merged_seq[30:36]
        stem2 = merged_seq[36:]
        
        full_seq = loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        full_seq_rna = full_seq.replace("T", "U")
        full_sequences.append(full_seq_rna)

        trigger = dna_reverse_complement_to_rna(switch)

        Ttigger_rna.append(trigger)

        on_seq =  pre_sequence + promoter + loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        off_sequence = pre_sequence + promoter + trigger + loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker

        On_sequences.append(on_seq )   
        Off_sequences.append(off_sequence)     

    return Off_sequences, On_sequences, full_sequences, Ttigger_rna




def process_toehold_structures(switch):

    Off_sequences, On_sequences, toehold_switch_sequence, Trigger_rnas = construct_toehold_trigger_from_list(merged_list=switch)

    MFE_ons = []
    MFE_offs = []

    DeltaDel_MFE=[]


    for off, on in zip(Off_sequences, On_sequences):

        structure, mfe_on = RNA.fold(on)
        structure, mfe_off = RNA.fold(off)

        MFE_ons.append(mfe_on)
        MFE_offs.append(mfe_off)

        DeltaDel_MFE.append(mfe_off-mfe_on)


    return Off_sequences, On_sequences, toehold_switch_sequence, Trigger_rnas, MFE_ons, MFE_offs, DeltaDel_MFE


def construct_toehold_from_list(merged_list):
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"

    full_sequences = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"Sequence length should be 45 nt, but found length {len(merged_seq)}: {merged_seq}")
        
        switch = merged_seq[:30]
        stem1 = merged_seq[30:36]
        stem2 = merged_seq[36:]
        
        full_seq = loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        full_sequences.append(full_seq)

    return full_sequences


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


def encode_structure_list(seqList):

    seqList = construct_toehold_from_list(merged_list=seqList)

    structures=np.array([build_structure_array(seq=mRNA) for mRNA in seqList])
    X_seq=np.array([Dimer_split_seqs(sequence) for sequence in seqList])

    return X_seq, structures



def compute_scaler(model_output):

    model_output = model_output.detach().cpu().numpy()

    return model_output

def create_argparser(promoters_number):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=promoters_number, device=device, schedule_low=1e-4,
    schedule_high=0.02,out_init_conv_padding = 1)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser


def main_function(promoters_number, opt=False, kind='radio'):



    args = create_argparser(promoters_number=promoters_number).parse_args()

    model_path = '../result/Switch-ddpm-2025-12-08-22-26-iteration-400-model.pth'
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    diffusion.load_state_dict(torch.load(model_path, weights_only=False))

    USE_RNA_ERNIE = True
    VOCAB = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/data/vocab/vocab_1MER.txt"
    ERNIE_PATH = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/code/CatIIIIIIII-RNAErnie-faa2b2d/output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final"
    ON_MODEL = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/model/adjusted_on_pearson_mse_structure_9_pcc=0.8279.pth'
    OFF_MODEL = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/model/off_pearson_mse_ernie_structure_12_pcc=0.7123.pth'

    predictor = ToeholdPredictorSimplified(
        on_model_path=ON_MODEL,
        off_model_path=OFF_MODEL,
        use_rna_ernie=USE_RNA_ERNIE,
        vocab_path=VOCAB,
        ernie_model_path=ERNIE_PATH,
        auto_select_gpu=False,
        device_num=1
    )


    sequences = []

    samples = diffusion.sample(args.num_images, device)
    samples = samples.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):

        decoded_sequence = decode_one_hot(samples[j])
        sequences.append("A" + decoded_sequence)


    full_sequences = construct_toehold_from_list(sequences)

    ons = []
    offs = []
    for seq in full_sequences:
        result = predictor.predict_single(seq)
        ons.append(result['ON'])
        offs.append(result['OFF'])

    ons = np.array(ons)
    offs = np.array(offs)


    print('offs = ', offs)
    
    if opt:
        all_sequences, all_ons, all_offs, all_radios = optimize_tensor_input_with_cmaes(diffusion_model = diffusion, predictor = predictor, number=promoters_number, 
                                     sigma=0.5, max_iter=30, popsize=promoters_number,kind = kind)
        return all_ons, all_offs, all_sequences
    else:
        final_ons = ons.tolist() if hasattr(ons, 'tolist') else ons
        final_offs = offs.tolist() if hasattr(offs, 'tolist') else offs
        final_seqs = sequences.tolist() if hasattr(sequences, 'tolist') else sequences
        
        return final_ons, final_offs, final_seqs




def blackbox_objective_z_tensor(z_flat_np, diffusion_model, predictor, shape=(512, 1, 4, 44), kind = 'difference'):


    sequences = []
    z_tensor = torch.tensor(z_flat_np, dtype=torch.float32).reshape(shape).to(device)

    samples = diffusion_model.sample_opt(x=z_tensor,batch_size=shape[0], device=device)

    samples = samples.squeeze(dim=1)

    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):

        decoded_sequence = decode_one_hot(samples[j])
        sequences.append("A" + decoded_sequence)


    Off_sequences, On_sequences, toehold_switch_sequence, Trigger_rnas, MFE_ons, MFE_offs, DeltaDel_MFE = process_toehold_structures(switch=sequences)

    

    full_sequences = construct_toehold_from_list(sequences)

    ons = []
    offs = []
    for seq in full_sequences:
        result = predictor.predict_single(seq)
        ons.append(result['ON'])
        offs.append(result['OFF'])

    ons = np.asarray(ons)
    offs = np.asarray(offs)


    if kind == 'radio':

        offs_safe = np.where(offs == 0, 1e-8, offs)
        radios = ons / offs_safe

        

    elif kind == 'difference':
        radios = ons - offs
        


    mean_radio = np.nanmean(radios)


    GC_ons = np.asarray([gc_content(on_seq) for on_seq in On_sequences], dtype=float)

    MFE_ons  = np.asarray(MFE_ons, dtype=float)
    MFE_offs = np.asarray(MFE_offs, dtype=float)

    pen_gc  = ITEM_PEN_MAX * float(np.mean(bounded_violation(GC_ons,   GC_ON_RANGE[0],   GC_ON_RANGE[1])))
    pen_on  = ITEM_PEN_MAX * float(np.mean(bounded_violation(MFE_ons,  MFE_ON_RANGE[0],  MFE_ON_RANGE[1])))
    pen_off = ITEM_PEN_MAX * float(np.mean(bounded_violation(MFE_offs, MFE_OFF_RANGE[0], MFE_OFF_RANGE[1])))

    penalty = pen_gc + pen_on + pen_off
    penalty = min(penalty, TOTAL_PEN_MAX)

    mean_radio = mean_radio + penalty


    return -mean_radio, sequences, ons, offs, radios, toehold_switch_sequence, Trigger_rnas, Off_sequences, On_sequences, MFE_ons, MFE_offs, DeltaDel_MFE


def tensor_to_promoter(output_tensor):
    
    samples = output_tensor.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()
    sequences = []
    
    for i in range(samples.shape[0]):

                    decoded_sequence = decode_one_hot(samples[i])
                    sequences.append(decoded_sequence)
                    
    return sequences

def prediction_strength(prediction_model, promoter):
    features = [np.array(Dimer_split_seqs(seq)) for seq in promoter]
    features = np.array(features)

    encoded_sequences = torch.tensor(features, dtype=torch.float32).to(device)

    raw_scores = prediction_model(encoded_sequences)
    
    min_strength = -8.6382
    max_strength = 12.5883
    raw_scores = raw_scores.squeeze().cpu().numpy()

    transformed_scores = 2 ** (raw_scores * (max_strength - min_strength) + min_strength)

    avg_strength = transformed_scores.mean()
    print('avg_strength = ',avg_strength)

    return avg_strength


def optimize_tensor_input_with_cmaes(diffusion_model, predictor, number=512, 
                                     sigma=0.5, max_iter=5, popsize=512, kind = 'difference'):
    shape=(number, 1, 4, 44)
    x = torch.randn(number, 1, 4, 44, device=device)
    x_flat = x.flatten().cpu()
    x = np.array(x_flat)

    es = cma.CMAEvolutionStrategy(x, sigma, {'popsize': popsize})

    best_score = -float('inf')
    best_z = None
    best_seq = None

    all_sequences = []
    all_ons = []
    all_offs = []
    all_radios = []

    for gen in range(max_iter):

        solutions = es.ask()
        scores = []


        for s in solutions:
        

            score, sequences, ons, offs, radios, toehold_switch_sequence, Trigger_rnas, Off_sequences, On_sequences, MFE_ons, MFE_offs, DeltaDel_MFE   = blackbox_objective_z_tensor(s, diffusion_model, predictor, shape=shape, kind = kind)

            df = pd.DataFrame({
                'on': ons,
                'off': offs,
                'radio': radios,
                'sequence': sequences,
                'toehold_switch_sequence': toehold_switch_sequence,
                'OFF_sequence': Off_sequences,
                'ON_sequence': On_sequences,
                'Trigger_rnas': Trigger_rnas,
                'DeltaDel_MFE': DeltaDel_MFE,
                'MFE_ON':MFE_ons,
                'MFE_OFF':MFE_offs,
                })
            

            today = datetime.now().strftime("%m%d")

            result_dir = f"../result-{kind}-{today}/"
            os.makedirs(result_dir, exist_ok=True)

            df.to_csv(os.path.join(result_dir, f"output_opt_iter={gen}_score={score}.csv"), index=False)


            
            all_sequences += sequences
            all_offs += offs.tolist()
            all_ons += ons.tolist()
            all_radios += radios.tolist()

            
            scores.append(score)

            if -score > best_score:

                best_score = -score


        es.tell(solutions, scores)
        print(f"[Gen {gen}] best average score = {best_score:.4f}")


    return all_sequences, all_ons, all_offs, all_radios


def divide_lists(ons: list, offs: list) -> list:
    if len(ons) != len(offs):
        raise ValueError("Lengths of the two lists are inconsistent")
    
    result = []
    for on, off in zip(ons, offs):
        if off == 0:
            result.append(float('inf'))
        else:
            result.append(on / off)
    
    return result

device = torch.device("cuda:1")
torch.cuda.set_device(1)


if __name__ == '__main__':

    opt = False
    promoter_number = 32
    ons = []
    sequences = []
    offs = []

    kind='difference'

    on, off, sequence = main_function(promoters_number=promoter_number, opt=opt, kind=kind)

    ons += on
    offs += off
    sequences += sequence

    df = pd.DataFrame({
    'on': on,
    'off':off,
    'radio':np.array(on, dtype=float)-np.array(off, dtype=float),
    'sequence': sequences
    })

    today = datetime.now().strftime("%m%d")

    result_dir = f"../result-{kind}-{today}/"

    if opt:
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number)
        df_sorted.to_csv(result_dir + 'opt_output.csv', index=False)
         
    else:
        
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number*3)
        df_sorted.to_csv(result_dir + 'output.csv', index=False)
