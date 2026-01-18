from Bio import SeqIO
from utils import *
import matplotlib.pyplot as plt
import pdb
import matplotlib as mpl
from matplotlib import rcParams

def get_promoter_by_fasta_file(file_name):
    sequences = []
    with open(file_name, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequences.append(str(record.seq))
    
    print(sequences[0])
    print('Number of generated promoters:', len(sequences))

    return sequences

if __name__ == '__main__':
    with open('../data/promoter.txt', 'r') as file:
        nat_sequences = [line.rstrip() for line in file]

    nat_number = len(nat_sequences)
    print('Natural promoter number is ', len(nat_sequences))

    train_number = int(0.9 * nat_number)

    ss1 = []
    ss2 = []
    corres = []
    epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]

    folder_path = '../sequences/'
    all_files = os.listdir(folder_path)
    file_names = []

    for file_name in all_files:
        if file_name.startswith('e'):
            file_names.append(file_name)
    
    for file_name in file_names:
        sequences = get_promoter_by_fasta_file(folder_path + file_name)

        for k in range(2, 7):
            s1, s2, corre = calculate_overall_kmer_correlation(dataset1=nat_sequences, dataset2=sequences, k=k, flag=True)

            s1 = s1.values
            s2 = s2.values

            ss1.append(s1)
            ss2.append(s2)
            corres.append(corre)

    rcParams['font.size'] = 18 

    print('Total number of subplots:', len(ss1))
    colors = [(78/255, 98/255, 171/255), (70/255, 158/255, 180/255), (135/255, 207/255, 164/255), (203/255, 233/255, 157/255), (253/255, 185/255, 106/255)]

    fig, axs = plt.subplots(17, 5, figsize=(30, 60))
    for i, (x, y, corre) in enumerate(zip(ss1, ss2, corres)):
        row = i // 5
        col = i % 5

        axs[row, col].scatter(x, y, c=colors[col])
        axs[row, col].set_title(f'PCC of {col+2}-mer: {round(corre, 3)}')
        axs[row, col].set_xlabel('Natural')
        axs[row, col].set_ylabel('Synthetic')

    plt.tight_layout()
    plt.savefig('../figure/k-mer correlationv2.svg')
    plt.show()
