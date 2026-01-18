import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from matplotlib import rcParams
import pdb

def compute_distance_between_motifs(sequences, motif1='TATAAT', motif2='TTGACA'):
    distances = []

    for seq in sequences:
        seq = seq.upper()
        pos_motif1 = seq.find(motif1)
        pos_motif2 = seq.find(motif2)

        if pos_motif2 == -1 or pos_motif1 == -1:
            continue

        dis = abs(pos_motif2 - pos_motif1) - len(motif1)
        distances.append(dis)
    
    return distances

def checkseq(seq, target, freq):
    checklist = []
    for j in range(50 - 6 + 1):
        check = 0
        for i in seq:
            i = i.upper()
            if i[j:j+6] == target:
                check += 1
        checklist.append(check / freq)
    return checklist

def plot_6_mer_many_model(ddpm_promoter, vae_promoter, path):
    natdata = np.load('../data/promoter.npy')
    nat_promoter = natdata.tolist()
    
    nat_promoter_number = len(nat_promoter)
    vae_promoter_number = len(vae_promoter)
    ddpm_promoter_number = len(ddpm_promoter)
    
    nat_TATAAT = checkseq(nat_promoter, 'TATAAT', nat_promoter_number)
    nat_TAAAAT = checkseq(nat_promoter, 'TAAAAT', nat_promoter_number)
    nat_ATTATT = checkseq(nat_promoter, 'ATTATT', nat_promoter_number)
    nat_TTTTTT = checkseq(nat_promoter, 'TTTTTT', nat_promoter_number)
    nat_AAAATG = checkseq(nat_promoter, 'AAAATG', nat_promoter_number)
    nat_AAAAAT = checkseq(nat_promoter, 'AAAAAT', nat_promoter_number)

    vae_TATAAT = checkseq(vae_promoter, 'TATAAT', vae_promoter_number)
    vae_TAAAAT = checkseq(vae_promoter, 'TAAAAT', vae_promoter_number)
    vae_ATTATT = checkseq(vae_promoter, 'ATTATT', vae_promoter_number)
    vae_TTTTTT = checkseq(vae_promoter, 'TTTTTT', vae_promoter_number)
    vae_AAAATG = checkseq(vae_promoter, 'AAAATG', vae_promoter_number)
    vae_AAAAAT = checkseq(vae_promoter, 'AAAAAT', vae_promoter_number)

    ddpm_TATAAT = checkseq(ddpm_promoter, 'TATAAT', ddpm_promoter_number)
    ddpm_TAAAAT = checkseq(ddpm_promoter, 'TAAAAT', ddpm_promoter_number)
    ddpm_ATTATT = checkseq(ddpm_promoter, 'ATTATT', ddpm_promoter_number)
    ddpm_TTTTTT = checkseq(ddpm_promoter, 'TTTTTT', ddpm_promoter_number)
    ddpm_AAAATG = checkseq(ddpm_promoter, 'AAAATG', ddpm_promoter_number)
    ddpm_AAAAAT = checkseq(ddpm_promoter, 'AAAAAT', ddpm_promoter_number)

    vae = np.vstack((vae_TATAAT, vae_TAAAAT, vae_ATTATT, vae_TTTTTT, vae_AAAATG, vae_AAAAAT))
    ddpm = np.vstack((ddpm_TATAAT, ddpm_TAAAAT, ddpm_ATTATT, ddpm_TTTTTT, ddpm_AAAATG, ddpm_AAAAAT))
    nat = np.vstack((nat_TATAAT, nat_TAAAAT, nat_ATTATT, nat_TTTTTT, nat_AAAATG, nat_AAAAAT))
        
    rcParams['font.size'] = 20
    fontsize = 22

    figure_name = ['TATAAT', 'TAAAAT', 'ATTATT', 'TTTTTT', 'AAAATG', 'AAAAAT']
    x = np.arange(-50, -5)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.set_facecolor('white')

    color1 = 'black'
    color2 = 'orange'
    color3 = 'red'

    for number in range(6):
        row = number // 3
        col = number % 3
        ax = axes[row, col]

        ax.plot(x, nat[number], color=color1, label='Natural', linewidth=3)
        ax.plot(x, vae[number], color=color2, label='VAE', linewidth=3)
        ax.plot(x, ddpm[number], color=color3, label='DDPM', linewidth=3)

        ax.set_yticks(np.arange(0.00, 0.14, 0.03))
        ax.tick_params(axis='y')
        ax.legend(fontsize=fontsize, loc='upper left')

        ax.set_title(figure_name[number])
        ax.set_xlabel("Distance to TSS")
        ax.set_ylabel(figure_name[number] + " Frequency")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_6_mer(sequences, path):
    natdata = np.load('../data/promoter.npy')
    gen_promoter = sequences
    nat_promoter = natdata.tolist()
    
    nat_promoter_number = len(nat_promoter)
    gen_promoter_number = len(gen_promoter)
    
    nat_TATAAT = checkseq(nat_promoter, 'TATAAT', nat_promoter_number)
    nat_TAAAAT = checkseq(nat_promoter, 'TAAAAT', nat_promoter_number)
    nat_ATTATT = checkseq(nat_promoter, 'ATTATT', nat_promoter_number)
    nat_TTTTTT = checkseq(nat_promoter, 'TTTTTT', nat_promoter_number)
    nat_AAAATG = checkseq(nat_promoter, 'AAAATG', nat_promoter_number)
    nat_AAAAAT = checkseq(nat_promoter, 'AAAAAT', nat_promoter_number)

    gen_TATAAT = checkseq(gen_promoter, 'TATAAT', gen_promoter_number)
    gen_TAAAAT = checkseq(gen_promoter, 'TAAAAT', gen_promoter_number)
    gen_ATTATT = checkseq(gen_promoter, 'ATTATT', gen_promoter_number)
    gen_TTTTTT = checkseq(gen_promoter, 'TTTTTT', gen_promoter_number)
    gen_AAAATG = checkseq(gen_promoter, 'AAAATG', gen_promoter_number)
    gen_AAAAAT = checkseq(gen_promoter, 'AAAAAT', gen_promoter_number)

    gen = np.vstack((gen_TATAAT, gen_TAAAAT, gen_ATTATT, gen_TTTTTT, gen_AAAATG, gen_AAAAAT))
    nat = np.vstack((nat_TATAAT, nat_TAAAAT, nat_ATTATT, nat_TTTTTT, nat_AAAATG, nat_AAAAAT))

    rcParams['font.size'] = 20
    fontsize = 22

    figure_name = ['TATAAT', 'TAAAAT', 'ATTATT', 'TTTTTT', 'AAAATG', 'AAAAAT']
    x = np.arange(-50, -5)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.set_facecolor('white')
    
    color1 = 'green'
    color2 = 'orange'

    for number in range(6):
        row = number // 3
        col = number % 3
        ax = axes[row, col]

        ax.plot(x, nat[number], color=color1, label='Natural', linewidth=3)
        ax.plot(x, gen[number], color=color2, label='Generated', linewidth=3)

        ax.set_yticks(np.arange(0.00, 0.034, 0.008))
        ax.tick_params(axis='y')

        ax.legend(fontsize=fontsize, loc='upper left')
        ax.set_title(figure_name[number])

        ax.set_xlabel("Distance to TSS")
        ax.set_ylabel(figure_name[number] + " Frequency")
    
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def get_promoter_by_fasta_file(file_name):
    sequences = []
    with open(file_name, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequences.append(str(record.seq))
    
    print(sequences[0])
    print('Number of generated promoters:', len(sequences))

    return sequences

if __name__ == '__main__':
    vae_promoter = get_promoter_by_fasta_file(file_name='../sequences/NUM_CHANNEL=256_RES_LAYERS=5_LATENT_DIM=10_BATCH_SIZE=1024_epoch=88_kernel=5_padding=2_corre=0.6988058182250327_gcfre=0.37324666666666667.fasta')
    ddpm_promoter = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1800_6_mer_fre_cor=0.8237605730640427.fasta')

    plot_6_mer_many_model(ddpm_promoter=ddpm_promoter, vae_promoter=vae_promoter, path='../figure/6_mer_frequence_VAE.pdf')

    mer_2 = [0.42735,0.98439,0.97749,0.96503,0.99230,0.96837,0.96484,0.95993,0.97056,0.98443,0.99159,0.99108,0.99435,0.99010,0.99749,0.99663,0.99337]
    mer_3 = [0.44366,0.95777,0.94357,0.94084,0.96908,0.93854,0.93618,0.93206,0.94688,0.96643,0.97694,0.98037,0.98300,0.98101,0.99057,0.99024,0.98859]
    mer_4 = [0.44847,0.93556,0.91565,0.92460,0.94573,0.91454,0.91351,0.91078,0.92791,0.94768,0.95687,0.96337,0.96571,0.96680,0.97581,0.97776,0.97695]
    mer_5 = [0.44038,0.90301,0.87833,0.89430,0.91206,0.88626,0.88277,0.88223,0.90253,0.92281,0.93123,0.93726,0.93893,0.94286,0.95175,0.95451,0.95625]
    mer_6 = [0.40673,0.83535,0.80837,0.82352,0.84080,0.83485,0.83101,0.83100,0.85396,0.86859,0.87704,0.88188,0.88268,0.88897,0.89677,0.89686,0.90457]

    rcParams['font.size'] = 26
    fontsize = 26

    plt.figure(figsize=(10, 8))
    epochs = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700]

    color1 = (183/255, 181/255, 160/255)
    color2 = (68/255, 117/255, 122/255)
    color3 = (69/255, 42/255, 61/255)
    color4 = (212/255, 76/255, 60/255)
    color5 = (221/255, 108/255, 76/255)

    linewidth = 3

    plt.plot(epochs, mer_2, label="2-mer", color=color1, linestyle='dashed', linewidth=linewidth, marker='D')
    plt.plot(epochs, mer_3, label="3-mer", color=color2, linewidth=linewidth, marker='s')
    plt.plot(epochs, mer_4, label="4-mer", color=color3, linestyle='dashed', linewidth=linewidth, marker='*')
    plt.plot(epochs, mer_5, label="5-mer", color=color4, linewidth=linewidth, marker='h')
    plt.plot(epochs, mer_6, label="6-mer", color=color5, linestyle='dashed', linewidth=linewidth, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Pearson correlation coefficient")
    plt.legend(fontsize=fontsize)

    plt.savefig('../figure/k_mer_pearson_in_different_epoch.pdf')
    plt.savefig('../figure/k_mer_pearson_in_different_epoch.svg')
    plt.show()
    plt.close()

    test_loss = np.load('../ddpm_logs/Not_all_promoter-ddpm-2024-09-02-16-35-iteration-300--kernel=3--test_loss.npy')
    x = np.linspace(1, 1800, 300, dtype=int)
    plt.figure(figsize=(10, 8))
    plt.plot(x, test_loss, label="test loss", color=color1, linewidth=linewidth)
    plt.xlabel("Epoch")
    plt.ylabel("Loss function value in test dataset")
    plt.legend(fontsize=fontsize)

    plt.savefig('../figure/test_loss.pdf')
    plt.savefig('../figure/test_loss.svg')
    plt.show()
    plt.close()

    pdb.set_trace()

    plt.figure(figsize=(4.5, 3))
    k_mer = [0.99337, 0.98859, 0.97695, 0.95625, 0.90457]
    bar_name = ['2-mer', '3-mer', '4-mer', '5-mer', '6-mer']
    colors_bar = [(130/255, 178/255, 154/255)] * 5

    plt.bar(bar_name, k_mer, color=colors_bar, width=0.4)
    plt.ylabel('pearson')
    plt.savefig('../figure/k_mer_in_same_epoch.pdf')
    plt.show()

    syn_promoter = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1800_6_mer_fre_cor=0.8237605730640427.fasta')
    plot_6_mer(sequences=syn_promoter, path='../figure/6_mer_frequence.pdf')

    pdb.set_trace()

    syn_promoter_1800 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1800_6_mer_fre_cor=0.8237605730640427.fasta')
    syn_promoter_1600 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1600_6_mer_fre_cor=0.818515167882261.fasta')
    syn_promoter_1400 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1400_6_mer_fre_cor=0.8204618326236955.fasta')
    syn_promoter_1200 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1200_6_mer_fre_cor=0.8134791657350415.fasta')
    syn_promoter_1000 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-1000_6_mer_fre_cor=0.814927075678317.fasta')
    syn_promoter_800 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-800_6_mer_fre_cor=0.8055995884401117.fasta')
    syn_promoter_600 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-600_6_mer_fre_cor=0.7966758949399824.fasta')
    syn_promoter_400 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-400_6_mer_fre_cor=0.7857123983374529.fasta')
    syn_promoter_200 = get_promoter_by_fasta_file(file_name='../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-200_6_mer_fre_cor=0.7389591403937376.fasta')
    
    natdata = np.load('../data/promoter.npy')
    nat_promoter = natdata.tolist()

    nat_promoter_distance = compute_distance_between_motifs(sequences=nat_promoter)
    syn_promoter_200_distance = compute_distance_between_motifs(sequences=syn_promoter_200)
    syn_promoter_400_distance = compute_distance_between_motifs(sequences=syn_promoter_400)
    syn_promoter_600_distance = compute_distance_between_motifs(sequences=syn_promoter_600)
    syn_promoter_800_distance = compute_distance_between_motifs(sequences=syn_promoter_800)
    syn_promoter_1000_distance = compute_distance_between_motifs(sequences=syn_promoter_1000)
    syn_promoter_1200_distance = compute_distance_between_motifs(sequences=syn_promoter_1200)
    syn_promoter_1400_distance = compute_distance_between_motifs(sequences=syn_promoter_1400)
    syn_promoter_1600_distance = compute_distance_between_motifs(sequences=syn_promoter_1600)
    syn_promoter_1800_distance = compute_distance_between_motifs(sequences=syn_promoter_1800)

    distances = [syn_promoter_200_distance, syn_promoter_400_distance, syn_promoter_600_distance,
                 syn_promoter_800_distance, syn_promoter_1000_distance, syn_promoter_1200_distance,
                 syn_promoter_1400_distance, syn_promoter_1600_distance, syn_promoter_1800_distance,
                 nat_promoter_distance]

    plt.figure(figsize=(10, 6))
    plt.boxplot(distances)
    plt.xticks(ticks=range(1, 11), labels=['200', '400', '600', '800', '1000', '1200', '1400', '1600', '1800', 'Natural'])
    plt.xlabel('Promoter Distances')
    plt.ylabel('Distances')

    plt.savefig('../figure/distance.pdf')
    plt.show()
