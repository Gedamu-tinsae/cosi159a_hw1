import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import argparse
import random
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
import os
import sys

# Additional imports from provided script
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='COSI 159A CIFAR-10 Training')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_best', action='store_true', help='resume from best model')
    parser.add_argument('--output_dir', default='models', type=str, help='Directory to save models and logs')
    parser.add_argument('--quiet', action='store_true', help='Reduce printed output')
    parser.add_argument('--config', type=str, help='Path to JSON or YAML config file')
    parser.add_argument('--early_stop', type=int, default=0, help='Early stopping patience (0 disables)')
    return parser.parse_args()

def main():
    args = get_args()
    # Config file support
    if args.config:
        if args.config.endswith('.json'):
            with open(args.config, 'r') as f:
                config = json.load(f)
        elif args.config.endswith(('.yaml', '.yml')):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError('Config file must be .json or .yaml')
        for k, v in config.items():
            setattr(args, k, v)
    
    # --- FOOLPROOF CHECK 1: Directory Management ---
    if not os.path.exists(args.output_dir):
        if not args.quiet:
            print(f"==> Warning: '{args.output_dir}' directory not found. Creating it now...")
        os.makedirs(args.output_dir)

    # --- FOOLPROOF CHECK 2: Dataset Presence ---
    if not os.path.exists('./data/cifar-10-batches-py'):
        print("!! ERROR: CIFAR-10 data not found in ./data/ !!")
        print("Please run 'python3 scripts/setup_data.py' first.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # --- FOOLPROOF CHECK 3: GPU Availability ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.quiet:
        print("==> Device Summary:")
        print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Seed: {args.seed}")
        print(f"  Output dir: {args.output_dir}")
    if device != 'cuda' and not args.quiet:
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
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')
    best_model_path = os.path.join(args.output_dir, 'model.pt')

    if args.resume_best:
        if os.path.isfile(best_model_path):
            if not args.quiet:
                print(f"==> Resuming from best model: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            print(f"!! Warning: No best model found at {best_model_path}. Starting from scratch !!")
    elif args.resume:
        if os.path.isfile(checkpoint_path):
            if not args.quiet:
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
    

    writer = SummaryWriter(f'runs/cifar10_lr{args.lr}_bs{args.batch_size}')
    train_losses, val_accuracies = [], []
    patience = args.early_stop
    best_epoch = 0
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loss, correct, total = 0, 0, 0

            loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False, desc=f"Epoch [{epoch+1}/{args.epochs}]", disable=args.quiet)
            for batch_idx, (inputs, targets) in loop:
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
                if not args.quiet:
                    loop.set_postfix(loss=train_loss/(batch_idx+1), acc=100.*correct/total)

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
            avg_train_loss = train_loss / len(trainloader)
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            # Log to TensorBoard
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            # Log learning rate
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            if not args.quiet:
                print(f'Epoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]["lr"]:.6f} | Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%')

            # Save Checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, checkpoint_path)

            # Save Best Weights
            if val_acc > best_val_acc:
                if not args.quiet:
                    print(f"*** New Best Validation Accuracy: {val_acc:.2f}%. Saving model... ***")
                torch.save(model.state_dict(), best_model_path)
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
            else:
                if patience > 0:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if not args.quiet:
                            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                        break

            scheduler.step()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        torch.save(model.state_dict(), best_model_path)
    writer.close()

    # Save curves as images
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'train_loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'val_accuracy_curve.png'))
    plt.close()

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
        avg_train_loss = train_loss / len(trainloader)
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

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

    writer.close()

    # --- 6. FINAL TEST EVALUATION ---
    if os.path.isfile(best_model_path):
        print("\n==> Training Complete. Evaluating BEST model on Test Set...")
        model.load_state_dict(torch.load(best_model_path))
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
        print(f"!! Error: {best_model_path} not found. Final evaluation skipped !!")

if __name__ == "__main__":
    main()