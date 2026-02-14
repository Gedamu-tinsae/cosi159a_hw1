import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import argparse
import os
import sys

def get_args():
    parser = argparse.ArgumentParser(description='COSI 159A CIFAR-10 Training')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- FOOLPROOF CHECK 1: Directory Management ---
    if not os.path.exists('models'):
        print("==> Warning: 'models' directory not found. Creating it now...")
        os.makedirs('models')

    # --- FOOLPROOF CHECK 2: Dataset Presence ---
    if not os.path.exists('./data/cifar-10-batches-py'):
        print("!! ERROR: CIFAR-10 data not found in ./data/ !!")
        print("Please run 'python3 scripts/setup_data.py' first.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    
    # --- FOOLPROOF CHECK 3: GPU Availability ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"==> Success: Training on GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("!! WARNING: CUDA not detected. Training on CPU will be extremely slow !!")

    # --- 1. DATA PREPARATION ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("==> Loading datasets...")
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_size = 45000
    val_size = 5000
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- 2. MODEL, LOSS, OPTIMIZER ---
    model = torchvision.models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 3. RESUME LOGIC ---
    start_epoch = 0
    best_val_acc = 0
    checkpoint_path = 'models/checkpoint.pt'

    if args.resume:
        if os.path.isfile(checkpoint_path):
            print(f"==> Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint.get('best_val_acc', 0)
            for _ in range(start_epoch):
                scheduler.step()
        else:
            print(f"!! Warning: No checkpoint found at {checkpoint_path}. Starting from scratch !!")

    # --- 4. TRAINING LOOP ---
    print(f"==> Starting training from Epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # --- 5. VALIDATION PHASE ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}/{args.epochs} | Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%')

        # Save Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, checkpoint_path)

        # Save Best Weights
        if val_acc > best_val_acc:
            print(f"*** New Best Validation Accuracy: {val_acc:.2f}%. Saving model... ***")
            torch.save(model.state_dict(), 'models/model.pt')
            best_val_acc = val_acc

        scheduler.step()

    # --- 6. FINAL TEST EVALUATION ---
    if os.path.isfile('models/model.pt'):
        print("\n==> Training Complete. Evaluating BEST model on Test Set...")
        model.load_state_dict(torch.load('models/model.pt'))
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        print(f'Final Test Accuracy: {100.*test_correct/test_total:.2f}%')
    else:
        print("!! Error: models/model.pt not found. Final evaluation skipped !!")

if __name__ == "__main__":
    main()