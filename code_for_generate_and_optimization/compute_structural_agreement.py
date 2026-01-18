from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def dotbracket_to_vector(structure: str) -> np.ndarray:
    mapping = {'.': 0, '(': 1, ')': 2}
    return np.fromiter((mapping[c] for c in structure.strip()), dtype=np.int8)


def compare_structures_max_similarity_fast(file_a: Union[str, Path],
                                          file_b: Union[str, Path]) -> float:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    if "Structure" not in df_a.columns or "Structure" not in df_b.columns:
        raise ValueError("Missing 'Structure' column")

    structures_a = df_a["Structure"].dropna().astype(str).tolist()
    structures_b = df_b["Structure"].dropna().astype(str).tolist()

    if not structures_a or not structures_b:
        raise ValueError("Structure column is empty")

    vecs_a = [dotbracket_to_vector(s) for s in structures_a]
    vecs_b = [dotbracket_to_vector(s) for s in structures_b]

    la = {len(v) for v in vecs_a}
    lb = {len(v) for v in vecs_b}
    if len(la) != 1 or len(lb) != 1 or next(iter(la)) != next(iter(lb)):
        raise ValueError("Inconsistent structure lengths")

    L = next(iter(la))
    vecs_a = np.vstack(vecs_a)
    vecs_b = np.vstack(vecs_b)

    matches = vecs_a[:, None, :] == vecs_b[None, :, :]
    similarities = matches.sum(axis=2) / L
    return float(similarities.max(axis=1).mean())


if __name__ == "__main__":
    folder = Path("../sequence_csv")
    nature = folder / "Nature_toehold_structures.csv"

    if not nature.exists():
        alt = Path("Nature_toehold_structures.csv")
        if alt.exists():
            nature = alt
        else:
            raise FileNotFoundError("Nature_toehold_structures.csv not found")

    csv_files = sorted(folder.glob("*.csv"))

    results = []
    for f in csv_files:
        if f.name == nature.name:
            continue
        try:
            agreement = compare_structures_max_similarity_fast(nature, f)
            results.append((f.name, agreement))
            print(f"[OK] {f.name} -> agreement={agreement:.6f}")
        except Exception as e:
            print(f"[SKIP] {f.name} -> {type(e).__name__}: {e}")

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\n===== agreement ranking (high->low) =====")
        for name, ag in results:
            print(f"{ag:.6f}\t{name}")
    else:
        print("No files successfully calculated agreement")

