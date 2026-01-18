import os
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import RNA
from rna_switch_energy import analyze_rna_switch

def construct_toehold_from_list(merged_list):
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"

    full_sequences = []
    Ttigger_rna = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"Sequence length should be 45 nt, but found {len(merged_seq)}: {merged_seq}")

        switch = merged_seq[:30]
        stem1 = merged_seq[30:36]
        stem2 = merged_seq[36:]

        full_seq = loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        full_seq_rna = full_seq.replace("T", "U")
        full_sequences.append(full_seq_rna)

        Ttigger_rna.append(dna_reverse_complement_to_rna(switch))

    return full_sequences, Ttigger_rna


def get_switch_by_fasta_file(file_name):
    sequences = []
    with open(file_name, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequences.append(str(record.seq))

    print(f"[INFO] Reading {file_name}, sequence count: {len(sequences)}")
    return sequences


def dna_reverse_complement_to_rna(dna_seq: str) -> str:
    rc_seq = str(Seq(dna_seq.upper()).reverse_complement())
    return rc_seq.replace("T", "U")


def process_toehold_structures(fasta_file: str, output_csv: str):
    switch = get_switch_by_fasta_file(fasta_file)
    toehold_seqs, triggers = construct_toehold_from_list(switch)

    records = []

    for seq, trigger in zip(toehold_seqs, triggers):
        structure, mfe = RNA.fold(seq)
        res = analyze_rna_switch(switch_seq=seq, trigger_seq=trigger)

        records.append({
            "Sequence": seq,
            "Trigger rna": trigger,
            "Structure": structure,
            "MFE_self (kcal/mol)": round(res["MFE_self"], 2),
            "MFE_hybrid (kcal/mol)": round(res["MFE_hybrid"], 2),
            "DeltaDeltaG_open (kcal/mol)": round(res["DeltaDeltaG_open"], 2),
        })

    df = pd.DataFrame(records)
    df_sorted = df.sort_values(by="DeltaDeltaG_open (kcal/mol)", ascending=True)
    df_sorted.to_csv(output_csv, index=False)

    print(f"[OK] File generated: {output_csv}")


if __name__ == "__main__":
    fasta_dir = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/Synthesizing_mRNA/sequence/"
    out_dir = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/Synthesizing_mRNA/sequence_csv/"

    os.makedirs(out_dir, exist_ok=True)

    whitelist_fasta = [
        "embweight_0.1(3)_Switche_epoch=50_6_mer_fre_cor=0.8855057903675256.fasta",
        "embweight_0.1(3)_Switche_epoch=100_6_mer_fre_cor=0.876471787357951.fasta",
        "embweight_0.1(3)_Switche_epoch=150_6_mer_fre_cor=0.8963284346169391.fasta",
        "embweight_0.1(3)_Switche_epoch=200_6_mer_fre_cor=0.7173103930840445.fasta",
        "embweight_0.1(3)_Switche_epoch=250_6_mer_fre_cor=0.7825819951935682.fasta",
        "embweight_0.1(3)_Switche_epoch=300_6_mer_fre_cor=0.8222449244690777.fasta",
        "embweight_0.1(3)_Switche_epoch=350_6_mer_fre_cor=0.7840919912327093.fasta",
        "embweight_0.1(3)_Switche_epoch=400_6_mer_fre_cor=0.8408635671498529.fasta",
        "embweight_0.1(3)_Switche_epoch=450_6_mer_fre_cor=0.8047800190127936.fasta",
        "embweight_0.1(3)_Switche_epoch=500_6_mer_fre_cor=0.785757817395356.fasta",
    ]

    print(f"[INFO] Processing {len(whitelist_fasta)} specific fasta files")

    for fasta in whitelist_fasta:
        input_path = os.path.join(fasta_dir, fasta)
        output_name = fasta.replace(".fasta", ".csv")
        output_path = os.path.join(out_dir, output_name)

        print(f"[RUNNING] Processing: {fasta}")
        process_toehold_structures(input_path, output_path)
