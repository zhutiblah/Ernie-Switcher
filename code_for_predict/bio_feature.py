from typing import List, Dict
import numpy as np
from collections import Counter
import re

def extract_kmer_features(seq: str, k: int = 3) -> Dict[str, float]:
    kmer_counts = Counter([seq[i:i+k] for i in range(len(seq)-k+1)])
    total = sum(kmer_counts.values())
    all_kmers = [a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT'] if k == 3 else []
    features = {f'kmer_{k}_{kmer}': kmer_counts.get(kmer, 0)/total for kmer in all_kmers}
    return features

def extract_CKSNAP_features(seq: str, gap: int = 5) -> Dict[str, float]:
    pairs = [a + b for a in "ACGT" for b in "ACGT"]
    feature_dict = {}
    for g in range(gap + 1):
        pair_counts = {p: 0 for p in pairs}
        for i in range(len(seq) - g - 1):
            p = seq[i] + seq[i + g + 1]
            if p in pair_counts:
                pair_counts[p] += 1
        total = sum(pair_counts.values()) or 1
        for p in pairs:
            feature_dict[f'CKSNAP_{p}_gap{g}'] = pair_counts[p] / total
    return feature_dict

def extract_NCP_features(seq: str) -> Dict[str, float]:
    chemical = {
        'A': [1, 1, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0],
        'T': [0, 0, 1],
        'U': [0, 0, 1]
    }
    features = []
    for nt in seq:
        features.extend(chemical.get(nt.upper(), [0, 0, 0]))
    return {f'NCP_{i}': val for i, val in enumerate(features)}

def extract_PseEIIP_features(seq: str) -> Dict[str, float]:
    EIIP = {'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'U': 0.1335}
    base = 'ACGT'
    trimer_list = [a + b + c for a in base for b in base for c in base]
    freq = {tri: 0 for tri in trimer_list}
    for i in range(len(seq) - 2):
        tri = seq[i:i+3]
        if tri in freq:
            freq[tri] += 1
    total = sum(freq.values()) or 1
    for k in freq:
        freq[k] /= total
    feature_dict = {f'PseEIIP_{tri}': freq[tri] * sum(EIIP.get(nt, 0) for nt in tri) for tri in trimer_list}
    return feature_dict

def get_all_features(seq: str) -> Dict[str, float]:
    seq = seq.upper()
    assert len(seq) == 80 and re.fullmatch('[ACGTU]+', seq), "输入必须是80nt的碱基序列"
    features = {}
    features.update(extract_kmer_features(seq))
    features.update(extract_CKSNAP_features(seq))
    features.update(extract_NCP_features(seq))
    features.update(extract_PseEIIP_features(seq))
    return features

example_features = get_all_features("A" * 80)
print('example_features = ', example_features)

