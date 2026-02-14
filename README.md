# COSI 159A - Assignment 1: CIFAR-10 Classification

This repository contains a complete pipeline for training, validating, and evaluating a ResNet-18 classifier on the CIFAR-10 dataset using PyTorch. It is organized for reproducibility, automation, and ease of use on cloud or local GPU instances.

---

## 📂 File & Folder Structure

```
├── data/           # CIFAR-10 dataset (auto-downloaded)
├── models/         # Saved model checkpoints and best weights
├── notebooks/      # Jupyter notebooks for EDA, prototyping, visualization
├── notes/          # Assignment instructions, guides, and breakdowns
├── scripts/        # Utility scripts (data setup, plotting, analysis)
│   ├── setup_data.py
│   ├── plot_results.py
│   └── analyze_results.py
├── .gitignore
├── File-Tree.md
├── README.md
├── requirements.txt
└── train.py        # Main training script
```

---

## 🚀 Quick Start

### 1. (Recommended) Create a Conda Environment
```bash
conda create -n cosi159a python=3.11 -y
conda activate cosi159a
```

### 2. Install PyTorch with CUDA Support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the CIFAR-10 Dataset
```bash
python3 scripts/setup_data.py
```

### 5. Train the Model
```bash
python3 train.py --epochs 30 --lr 0.1 --batch_size 128
```

---

## Alternative: Using venv (for simple projects)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🏗️ Scripts & Usage

- **train.py**: Main training/validation/testing script. Supports resuming from checkpoints with `--resume`.
- **scripts/setup_data.py**: Downloads and prepares the CIFAR-10 dataset.
- **scripts/plot_results.py**: Example script for plotting training/validation accuracy curves (customize as needed).
- **scripts/analyze_results.py**: Example script for analyzing predictions and computing confidence intervals.

---

## 📒 Notebooks

Use the notebooks/ directory for Jupyter notebooks to:
- Explore and visualize the dataset (EDA)
- Prototype new ideas or debug code
- Plot results interactively

---

## 📝 Notes & Documentation

See the notes/ directory for:
- Assignment instructions and grading (Assignment_1.md, Assignment_1.pdf)
- Internal guides for running on cloud, using tmux, etc.
- Detailed breakdown of the file structure and requirements

---

## 💡 Tips

- Conda is preferred for deep learning and GPU projects.
- Use the --resume flag in train.py to continue interrupted training jobs.
- Use scripts/plot_results.py and scripts/analyze_results.py as templates for your own analysis.
- Keep notebooks/ for interactive work and scripts/ for automation.

---

## 📚 Requirements

All dependencies are listed in requirements.txt. Key libraries include:
- torch, torchvision (deep learning)
- numpy, scipy, pillow (data handling)
- matplotlib, tensorboard, tqdm (visualization & progress)

---

## 📬 Contact

For questions, see the notes/ directory or contact the course staff.