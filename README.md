# UNIFY
## Paper "UNIFY: Unified Co-Attention on Hierarchical Biological Networks for Enhanced Drug–Target Interaction Prediction"


## Repository Structure
- data/
  - Luo/
    - features.csv — Feature matrix for drugs and targets
    - positive_edge.csv — All positive samples (known drug–target pairs)
    - negative_edge.csv — All negative samples (unobserved drug–target pairs)
  - BioSNAP/
    - features.csv — Feature matrix for drugs and targets
    - positive_edge.csv — All positive samples (known drug–target pairs)
    - negative_edge.csv — All negative samples (unobserved drug–target pairs)

- main.py — Load the dataset and run DTI prediction
- model.py — UNIFY model implementation
- requirements.txt — Python dependencies

The data directory contains benchmark datasets and biological knowledge related to drugs and targets.

### Quick start

1. Create and activate a conda environment
```bash
conda create -n unify python=3.10
conda activate unify
```

2. Install dependencies
```bash
pip install requirements.txt -r
```
3. Run the example
```bash
python main.py
```



