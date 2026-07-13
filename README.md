# DGCMF-PLM-ACP

## Interpretable and Parameter-Efficient Protein Language Model Fusion for Anticancer Peptide Discovery

DGCMF-PLM-ACP is a dual-granularity cross-modal fusion framework for anticancer peptide (ACP) prediction. It combines residue identity, physicochemical descriptors, and contextual representations from a frozen protein language model while keeping the number of trainable parameters small.



## Overview

Existing ACP predictors often use protein language model embeddings as shallow additive features or fine-tune the entire language model. The former may underuse the hierarchical information encoded by protein language models, while the latter is computationally expensive and can overfit small ACP datasets.

DGCMF-PLM-ACP addresses these limitations through complementary fusion at two semantic granularities:

- **Sequence-level fusion:** globally pooled ESM-2 representations condition handcrafted feature maps through Feature-wise Linear Modulation (FiLM).
- **Residue-level fusion:** token-level ESM-2 representations interact with handcrafted features through multi-head cross-attention (MHCA).
- **Complementary sequence modeling:** parallel BiGRU and Transformer branches capture order-sensitive and global contextual patterns.
- **Adaptive integration:** a learnable gate combines the two branches before classification.
- **Parameter efficiency:** the ESM-2 backbone remains frozen; the downstream framework has approximately **1.06 million trainable parameters**, about **615× fewer** than the fully fine-tuned comparison model described in the study.

## Framework

![Architecture of the DGCMF-PLM-ACP framework](./DGCMF_PLM_ACP.pdf)

The framework uses three complementary input modalities:

1. **Binary Profile Feature (BPF):** preserves residue identity and sequence order.
2. **Z-scale descriptors:** encode physicochemical properties including hydrophobicity, steric bulk, polarity, polarizability, and electronic effects.
3. **ESM-2 embeddings:** provide contextual residue-level representations from the frozen `esm2_t33_650M_UR50D` model.

## Main Results

DGCMF-PLM-ACP was evaluated on four public benchmark datasets: **ACP740**, **ACP Main**, **ACP Alternate**, and **DeepGram**.

- On **ACP740**, the model achieved 89.59% accuracy, 94.37% specificity, 90.32% F1-score, and 79.86% MCC.
- It obtained the best reported MCC in the study on ACP740 and competitive performance on ACP Alternate, DeepGram, and ACP Main.
- Across datasets, the framework maintained a balanced sensitivity-specificity trade-off and strong false-positive control.
- Ablation studies identified ESM-2 representations as the dominant contextual signal, with BPF and Z-scale descriptors providing complementary information.
- Cross-attention saliency and t-SNE analyses indicated biologically plausible associations with charge-related and hydrophobic residues.

## Repository Structure

```text
.
├── checkpoints/
├── data/
├── models/
├── preprocess/
├── tools/
├── utils/
├── train.py
├── eval.py
├── requirements.txt
├── DGCMF_PLM_ACP.svg
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train a model:

```bash
python train.py --dataset ACP740
```

Evaluate a trained model:

```bash
python eval.py --dataset ACP740
```


## Datasets
The study uses processed versions of four previously published and publicly available ACP benchmarks:

- ACP740
- AntiCP 2.0 Main Dataset (ACP Main)
- AntiCP 2.0 Alternate Dataset (ACP Alternate)
- DeepGram/LEE Dataset (DeepGram)

The processed datasets used in the experiments are included in this repository for reproducibility.

## Checkpoints and Results

Large checkpoints and experiment outputs are stored separately:

- [Pretrained checkpoints][(https://drive.google.com/file/d/1wK-aUZu9OveWrraVFq7gZh_O50FCch6j/view?usp=drive_link)
- [Evaluation results, logs, metrics, and plots](https://drive.google.com/file/d/1rRbJsbRwCvzSO3-u_gXCm71bE1zVF_Xm/view?usp=drive_link)](https://drive.google.com/drive/folders/1fMYdQYJCO5vLVSXgcHj_SHMMZEDQ2JFu?usp=drive_link)

## Citation

The manuscript is currently under review. Citation information will be added after publication.

**Manuscript title:** *Interpretable and parameter-efficient protein language model fusion for anticancer peptide discovery*

## Contact

For questions about the study, please contact:

- YuZhang:zhangyuhrb04@gmail.com


