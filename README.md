# Stereotypes in Style: Gender Bias in Multi-Modal AI Models

**Master's Dissertation — University of Malta, 2025**  
**Author:** Charla Pia Vella  
**Supervisors:** Dr Dylan Seychell, Prof. Vanessa Camilleri

---

## Overview

This repository contains all code and results produced 
for the dissertation investigating gender bias in fashion 
AI across visual classification, interpretability analysis, 
and cross-modal evaluation using InstructBLIP.

---

## Key Findings

- InstructBLIP reproduces ResNet-50's incorrect male 
  attribution in **88%** of female misclassification 
  cases under neutral prompting
- Male subjects correctly identified in **92%** of cases 
  despite ResNet-50 misclassification
- The **Female Majority Paradox**: more female training 
  images does not produce better female classification
- Visual stereotypes propagate directly into linguistic 
  representations across architecturally unrelated models
- Internal stereotype contradictions identified within 
  single generated sentences in InstructBLIP outputs

---

## Repository Structure

stereotypes-in-style/
├── preprocessing/   — Face cropping and occlusion using MTCNN
├── training/        — ResNet-50 and EfficientNet-B0 training
├── models/          — FashionCLIP, ArcFace, fusion classifier
├── evaluation/      — Grad-CAM, InstructBLIP cross-modal analysis
├── results/         — Experimental outputs and CSV results
└── dissertation/    — Final dissertation PDF

---

## Dataset

Experiments were conducted on the **RichWear dataset**.  
The dataset is not included in this repository.  
Please refer to the original dataset paper for access:  
IEEE Automatic Control and Systems Engineering  
Street Photo Dataset (2021)

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Hardware

All experiments were conducted on an **NVIDIA RTX 4090 
GPU (24GB VRAM)** via the Vast.ai cloud computing 
platform with CUDA 12.8.

---

## Reproducing the Experiments

1. Download the RichWear dataset and place images 
   in a local directory
2. Update image paths in the relevant scripts to 
   point to your local dataset location
3. Run training scripts from the `training/` folder
4. Run evaluation scripts from the `evaluation/` folder
5. InstructBLIP requires a GPU with at least 16GB VRAM