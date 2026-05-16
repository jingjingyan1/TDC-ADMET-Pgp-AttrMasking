# Pgp_Broccatelli — AttrMasking GNN (AUROC 0.937 ± 0.004)

Submission to the [TDC ADMET Benchmark Leaderboard](https://tdcommons.ai/benchmark/admet_group/03pgp/) for P-glycoprotein (Pgp) inhibition prediction on the Pgp_Broccatelli dataset.

## Results

| Seed | AUROC | AUPRC | F1 |
|------|-------|-------|----|
| 1 | 0.943 | 0.951 | 0.876 |
| 2 | 0.930 | 0.946 | 0.856 |
| 3 | 0.937 | 0.946 | 0.859 |
| 4 | 0.937 | 0.944 | 0.881 |
| 5 | 0.939 | 0.947 | 0.861 |
| **Mean ± Std** | **0.937 ± 0.004** | **0.947 ± 0.002** | **0.867 ± 0.010** |

## Method

We fine-tune a pretrained Graph Isomorphism Network (GIN) using the attribute masking strategy (Hu et al., 2020) implemented in [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose). The GIN was pretrained via self-supervised learning on approximately 2 million molecules from the ZINC15 and ChEMBL datasets. The pretrained molecular encoder is then fine-tuned end-to-end on the Pgp_Broccatelli training set with an MLP classification head.

### Architecture
- **Molecular encoder**: 5-layer GIN pretrained with attribute masking (`gin_supervised_masking`)
- **Classification head**: MLP with hidden dimensions [512, 128]
- **Total parameters**: 2,067,053

### Training Configuration
- **Epochs**: 100
- **Learning rate**: 0.0005
- **Batch size**: 128
- **Optimizer**: Adam (DeepPurpose default)
- **Loss**: Binary cross-entropy

### Data
- **Dataset**: [Pgp_Broccatelli](https://tdcommons.ai/single_pred_tasks/adme/#pgp-p-glycoprotein-inhibition-broccatelli-et-al) (1,218 molecules)
- **Task**: Binary classification (Pgp substrate: yes/no)
- **Split**: Official TDC scaffold split (train: 851, valid: 122, test: 245)
- **Evaluation**: 5 independent runs with seeds [1, 2, 3, 4, 5]

## Reproduction Guide

### Tested Environment
| Component | Version |
|-----------|---------|
| GPU | NVIDIA A100 |
| Python | 3.12 |
| PyTorch | 2.4.0+cu121 |
| CUDA (PyTorch) | 12.1 |
| DGL | 2.4.0+cu124 |
| DGLLife | 0.3.2 |
| DeepPurpose | latest (pip) |
| PyTDC | latest (pip) |

### Step-by-step Instructions

**1. Install dependencies (run in a Colab cell):**
```bash
# Install PyTorch 2.4.0 with CUDA 12.1
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install DGL (use cu124 wheel; cu121 may return 403)
!pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Install DGLLife, DeepPurpose, and PyTDC
!pip install dgllife DeepPurpose pytdc

# Install DeepPurpose dependencies
!pip install git+https://github.com/bp-kelley/descriptastorus pandas-flavor
```

**2. Restart the runtime** (Runtime → Restart runtime). Installed packages persist after restart.

**3. Upload `pgp_gnn_colab.py` and run:**
```bash
!python pgp_gnn_colab.py 2>&1 | tee results.txt
```

Total runtime: approximately 15-20 minutes on T4, 10-15 minutes on A100.

### Troubleshooting
- **DGL 403 error**: The `cu121` wheel URL may be temporarily unavailable. Use `cu124` instead:
  ```bash
  !pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
  ```
- **`ModuleNotFoundError: No module named 'dgl'`**: Restart the runtime after installing DGL.
- **`libcudart.so.11.0` error**: You installed a CUDA 11.x DGL on a CUDA 12.x system. Reinstall with the cu124 wheel.
- **`No module named 'descriptastorus'`**: Run `!pip install git+https://github.com/bp-kelley/descriptastorus pandas-flavor`.

## File Structure
```
├── README.md               # This file
├── pgp_gnn_colab.py        # Training and evaluation script
├── requirements.txt        # Dependency list
└── results.txt             # Full output log from the run
```

## References

1. Hu, W., Liu, B., Gomes, J., et al. (2020). Strategies for Pre-training Graph Neural Networks. *ICLR 2020*. [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)
2. Huang, K., Fu, T., Glass, L.M., et al. (2021). DeepPurpose: a Deep Learning Library for Drug–Target Interaction Prediction. *Bioinformatics*, 36(22-23), 5545–5547. [DOI](https://doi.org/10.1093/bioinformatics/btaa1005)
3. Huang, K., Fu, T., Gao, W., et al. (2021). Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. *NeurIPS 2021 Datasets and Benchmarks*. [arXiv:2102.09548](https://arxiv.org/abs/2102.09548)
4. Broccatelli, F., Carosati, E., Neri, A., et al. (2011). A Novel Approach for Predicting P-Glycoprotein (ABCB1) Inhibition Using Molecular Interaction Fields. *J. Med. Chem.*, 54(6), 1740–1751.

## License

MIT
