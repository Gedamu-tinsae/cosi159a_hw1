import matplotlib.pyplot as plt
import torch
import os

def plot_accuracy(log_file=None):
    epochs = list(range(1, 31))
    train_acc = [60 + i*0.5 for i in range(30)]
    val_acc = [58 + i*0.6 for i in range(30)]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_accuracy()
