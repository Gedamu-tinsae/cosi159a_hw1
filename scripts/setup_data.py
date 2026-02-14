import torchvision
import os

def setup():
    data_path = './data'
    print(f"==> Checking for data in {data_path}...")
    
    # This downloads AND extracts the data
    torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
    
    if os.path.exists(os.path.join(data_path, 'cifar-10-batches-py')):
        print("==> Success: CIFAR-10 is ready for training.")
    else:
        print("!! Warning: Download finished but directory structure looks unexpected.")

if __name__ == "__main__":
    setup()