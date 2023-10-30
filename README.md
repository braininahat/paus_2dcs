# PAUS 2D Cross Section Classification

This repository contains code for the paper -

## asclepius

asclepius is a commandline tool to preprocess data, split into train/val/test, train models, and evaluate them.

### Installation

### Usage

#### Preprocess dataset

```bash
python main.py prepare-dataset
```

#### Train Val Test Split

```bash
python main.py train-val-test-split  --dataset-split-path dataset/preprocessed/bal --split-ratios 6,2,2
```

#### Training

```bash
python main.py trainer --model-names cnn,swin,vit,densenet --modalities US,PA,PAUS --target-size 224,224 --batch-size 16
```

#### Testing

```bash
python main.py tester --weights-root output/20231030103142 --target-size=224,224
```
