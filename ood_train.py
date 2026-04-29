import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import get_dataset
from models.cifar.densenet import DenseNet3


def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_options():
    parser = argparse.ArgumentParser()

    # similar to ood_eval.py
    parser.add_argument("--ind_dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="densenet", choices=["densenet"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # noisy data args, handled inside utils.dataset.get_dataset()
    parser.add_argument("--use_noisy_data", action="store_true")
    parser.add_argument("--noise_ratio", type=float, default=0.4)
    parser.add_argument("--noise_type", type=str, default="symmetric", choices=["symmetric", "asymmetric"])
    parser.add_argument("--noise_seed", type=int, default=42)

    # saving
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default=None)

    args = parser.parse_args()
    return args


def build_model(args):
    if args.ind_dataset == "cifar10":
        args.num_classes = 10
    elif args.ind_dataset == "cifar100":
        args.num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {args.ind_dataset}")

    if args.model == "densenet":
        model = DenseNet3(
            depth=100,
            num_classes=args.num_classes,
            growth_rate=12,
            reduction=0.5,
            bottleneck=True,
            dropRate=0.0,
        )
        return model

    raise ValueError(f"Unsupported model: {args.model}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def main():
    args = get_train_options()
    fix_random_seed(args.random_seed)

    if args.ind_dataset == "cifar10":
        args.num_classes = 10
    elif args.ind_dataset == "cifar100":
        args.num_classes = 100

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )

    print(args)
    print("device:", device)

    train_data, test_data = get_dataset(args.ind_dataset, args)

    train_loader = DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
        gamma=0.1,
    )

    if args.save_name is None:
        noise_tag = "clean"
        if args.use_noisy_data:
            noise_tag = f"{args.noise_type}_noise{int(args.noise_ratio * 100)}"

        args.save_name = f"{args.ind_dataset}_{args.model}_{noise_tag}.pth"

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, args.save_name)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={total_loss / total:.4f} | "
            f"train_acc={train_acc:.2f} | "
            f"test_acc={test_acc:.2f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    print(f"Best test acc: {best_acc:.2f}")
    print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
