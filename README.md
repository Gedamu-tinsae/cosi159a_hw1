# COSI 159A - Assignment 1: CIFAR-10 Classification

This repository contains the implementation of a ResNet-based classifier for the CIFAR-10 dataset.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python3 scripts/setup_data.py`

## How to Train
Run the training script using argparse:
```bash
python3 train.py --epochs 30 --lr 0.1 --batch_size 128