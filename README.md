
# RNA Toehold Switch Design via Diffusion Generation, Prediction and Biophysical Optimization

## Overview
This repository implements a complete computational pipeline for RNA toehold switch design, integrating:

- Diffusion-based generation of RNA sequences  
- Prediction models for ON / OFF translation strength  
- Biophysical optimization guided by RNA secondary structure and thermodynamics  

The code is organized to emphasize methodological clarity and reproducibility. This document is intended to help reviewers quickly understand the structure, purpose, and execution flow of the project.

## Overall Pipeline
```
Diffusion-based generation
        ↓
Prediction of ON / OFF translation strength
        ↓
Biophysical optimization (structure and energy guided)
```

## Project Structure
```
generation_optimization/
├── train_cifar.py
├── diffusion.py
├── unet.py
├── ema.py
├── script_utils.py
├── main_opt_biophysical.py
├── make_synthetic_toehold_switch.py
├── rna_switch_energy.py
├── utils.py
├── utils_extra.py
├── test_complete.py

prediction/
├── on_structure_test1.py
├── off_structure_test1.py
├── switch_predict.py
├── sequence_classification.py
├── api_server.py
├── testapi.py
├── bio_feature.py
├── base_classes.py
├── tokenizer_nuc.py
├── dataset_utils.py
├── visualizer.py

analysis_visualization/
├── compute_structural_agreement.py
├── plot_k_mer_corre.py
├── make_paper_figure.py
├── vis_data.py
```

## Generation Module: Diffusion Model
**Purpose**  
Learns a distribution over valid RNA toehold switch sequences using a denoising diffusion probabilistic model (DDPM).

**Main Entry**
- `train_cifar.py`

**Key Features**
- UNet-based diffusion model  
- Linear or cosine noise schedules  
- Exponential Moving Average (EMA) for stable sampling  
- Optional RNAErnie embedding alignment  
- Periodic RNA sequence generation  

## Prediction Module: Translation Strength
**Purpose**  
Maps a full-length RNA toehold switch sequence (115 nt) to:
- ON translation strength  
- OFF translation strength  
- ON/OFF ratio  

**Main Training Scripts**
- `on_structure_test1.py`
- `off_structure_test1.py`

**Prediction Interface**
- `switch_predict.py`  
  - Batch prediction support  
  - Optional RNAErnie embeddings  
  - Used by optimization module and API  

## Biophysical Optimization Module
**Purpose**  
Refines candidate RNA sequences using black-box optimization guided by biological constraints.

**Main Entry**
- `main_opt_biophysical.py`

**Optimization Method**
- CMA-ES

**Objectives**
- Predicted ON translation strength  
- Predicted OFF translation strength  
- ON/OFF ratio  
- GC content constraints  
- RNA secondary structure free energy (MFE)  
- ΔΔG between alternative structures  

### Key Utility Scripts
- `make_synthetic_toehold_switch.py`  
  Constructs fixed-length (115 nt) synthetic toehold switches from modular components.

- `rna_switch_energy.py`  
  Performs RNA secondary structure and thermodynamic analysis:
  - RNAfold / RNAcofold  
  - MFE computation  
  - ΔΔG calculation  
  - Structure parsing  

## RNAErnie Large Model
Optional integration of **RNAErnie**, a pretrained RNA language model:
- Used for prediction models  
- Used for diffusion embedding alignment  

Official repository: https://github.com/CatIIIIIIII/RNAErnie

### Embedding Cache
RNAErnie embeddings are generated on first run and cached in:
```
ernie_embeddings_batches/
```
This avoids repeated forward passes.

## Environment Setup
### System Requirements
- Linux (recommended)  
- NVIDIA GPU with CUDA  
- Python ≥ 3.8  

### Conda Environment
```bash
conda create -n rna_switch python=3.9
conda activate rna_switch
```

### Python Dependencies
```bash
pip install torch torchvision
pip install numpy pandas scipy tqdm
pip install matplotlib seaborn
pip install biopython
pip install cma
pip install fastapi uvicorn
```

### ViennaRNA (Required)
```bash
conda install -c bioconda viennarna
python -c "import RNA; print(RNA.__version__)"
```

### RNAErnie Dependencies (Optional)
```bash
pip install paddlepaddle-gpu
pip install paddlenlp
```

## Running the Pipeline
```bash
python train_cifar.py
python on_structure_test1.py
python off_structure_test1.py
python main_opt_biophysical.py
```

## Notes for Reviewers
- Datasets are not included due to size and licensing  
- Modules are clearly separated  
- Embeddings are cached for efficiency  
- Emphasis on methodological transparency  

## License
MIT License © 2024
