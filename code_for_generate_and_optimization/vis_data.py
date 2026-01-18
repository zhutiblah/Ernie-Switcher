import pandas as pd

def read_data(filename):

    df = pd.read_csv(filename)
    required_columns = ['loop1', 'switch', 'loop2', 'stem1', 'atg', 'stem2', 'linker', 'post_linker', 'ON', 'OFF', 'ON_OFF']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Error: Data missing column '{col}'")

    df = df.dropna(subset=['ON', 'OFF', 'ON_OFF'])

    ons = df['ON'].astype(float).tolist()
    offs = df['OFF'].astype(float).tolist()
    on_offs = df['ON_OFF'].astype(float).tolist()
    loop1s = df['loop1'].tolist()
    switches = df['switch'].tolist()
    loop2s = df['loop2'].tolist()
    stem1s = df['stem1'].tolist()
    atgs = df['atg'].tolist()
    stem2s = df['stem2'].tolist()
    linkers = df['linker'].tolist()
    post_linkers = df['post_linker'].tolist()

    mRNAs = [l1 + sw + l2 + s1 + atg + s2 + lk + pl for l1, sw, l2, s1, atg, s2, lk, pl in 
             zip(loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers)]

    constant_part = [l1 + atg + lk + pl for l1, atg, lk, pl in zip(loop1s, atgs, linkers, post_linkers)]
    variable_part = [sw + s1 + s2 for sw, s1, s2 in zip(switches, stem1s, stem2s)]
    

    print(f'Reading complete, total {len(mRNAs)} entries')

    return mRNAs, constant_part, variable_part, loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers, ons, offs, on_offs

def check_consistency(*sequences):
    result = {}
    
    for i, seq_list in enumerate(sequences):
        unique_values = set(seq_list)
        
        if len(unique_values) == 1:
            result[f"sequence_{i+1}"] = list(unique_values)[0]
        else:
            result[f"sequence_{i+1}"] = None
            
    return result

def save_mrna_to_fasta(mrna_list, filename):
    with open(filename, 'w') as file:
        for i, mrna in enumerate(mrna_list):
            file.write(f">sequence_{i+1}\n")
            file.write(mrna + "\n")

    print(f"FASTA file saved to {filename}")


if __name__ == '__main__':
 
    filename = '/home/liangce/lx/Promoter_mRNA_synthetic/data/Toehold_mRNA_Dataset_clean.csv'
    mRNAs, constant_part, variable_part, loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers, ons, offs, on_offs = read_data(filename=filename)

    save_mrna_to_fasta(mRNAs, '../data/mrna_sequences.fasta')
    save_mrna_to_fasta(constant_part, '../data/constant_part.fasta')
    save_mrna_to_fasta(variable_part, '../data/variable_part.fasta')

    result = check_consistency(mRNAs, loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers)

    for key, value in result.items():
        print(f"{key}: {value}")

    print('Results show that Switch, stem1 and stem2 are variable, others are constant.')

    print(f'Variable part lengths: stem1:{len(stem1s[0])}; stem2:{len(stem2s[0])}; Switch:{len(switches[0])}')
