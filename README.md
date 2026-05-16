# Pgp_Broccatelli - AttrMasking GNN

Submission to the [TDC ADMET Benchmark Leaderboard](https://tdcommons.ai/benchmark/admet_group/03pgp/) for the Pgp_Broccatelli dataset.

## Method

We use a pretrained Graph Isomorphism Network (GIN) with attribute masking (`DGL_GIN_AttrMasking`) from the [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) toolkit. The model was pretrained on ~2M molecules using a self-supervised attribute masking strategy (Hu et al., 2020), then fine-tuned on the Pgp_Broccatelli training set.

### Model Details
- **Architecture**: GIN (Graph Isomorphism Network) with 5 layers
- **Pretraining**: Supervised attribute masking on ~2M molecules
- **Fine-tuning**: 100 epochs, learning rate 0.0005, batch size 128
- **Classifier head**: MLP with hidden dimensions [512, 128]
- **Framework**: DeepPurpose + DGL + DGLLife

## Results

| Dataset | Metric | Result |
|---------|--------|--------|
| Pgp_Broccatelli | AUROC | **0.937 ± 0.004** |

Results are averaged over 5 independent runs with different random seeds (1-5), using the official TDC benchmark splits.

## How to Reproduce

### Requirements
- Python 3.12
- PyTorch 2.4.0 (CUDA 12.1)
- DGL 2.1.0
- DGLLife 0.3.2
- DeepPurpose
- PyTDC

### Installation
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install dgllife DeepPurpose pytdc
pip install git+https://github.com/bp-kelley/descriptastorus pandas-flavor
```

### Run
```bash
python pgp_gnn_colab.py
```

Tested on Google Colab with NVIDIA A100 GPU.

## References

- Hu, W., et al. (2020). Strategies for Pre-training Graph Neural Networks. ICLR 2020. [Paper](https://arxiv.org/abs/1905.12265)
- Huang, K., et al. (2021). DeepPurpose: a Deep Learning Library for Drug-Target Interaction Prediction. Bioinformatics. [Paper](https://doi.org/10.1093/bioinformatics/btaa1005)
- Huang, K., et al. (2021). Therapeutics Data Commons. NeurIPS 2021. [Paper](https://arxiv.org/abs/2102.09548)
