import RNA

def analyze_rna_switch(switch_seq, trigger_seq, temperature=37):
    results = {}

    md = RNA.md()
    md.temperature = temperature
    fc_self = RNA.fold_compound(switch_seq, md)
    structure, mfe_self = fc_self.mfe()
    results["MFE_self"] = mfe_self
    results["Structure_self"] = structure
    hybrid_seq = switch_seq + '&' + trigger_seq
    structure_hybrid, mfe_hybrid = RNA.cofold(hybrid_seq)
    results["MFE_hybrid"] = mfe_hybrid
    results["Structure_hybrid"] = structure_hybrid
    results["DeltaDeltaG_open"] = mfe_self - mfe_hybrid

    return results

if __name__ == "__main__":
    
    switch_seq = "AUGCAUUGAUGCUACGGAUACGUAGCUAUGCAUUGAUGCUACGGAU"
    trigger_seq = "UACGUACCUAGCUAUGCUA"

    res = analyze_rna_switch(switch_seq, trigger_seq)
    for k, v in res.items():
        print(f"{k}: {v}")
