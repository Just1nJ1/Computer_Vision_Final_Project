import csv
import warnings
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_model(args):
    task = args.task
    honeypot_pos = args.honeypot_pos
    h_factor = args.h_factor
    if task == "Honeypot":
        assert honeypot_pos is not None, "Missing honeypot position"
        from HoneypotModel import VGG16Modified, train
        model = VGG16Modified(num_classes=args.num_classes, honeypot_pos=honeypot_pos)
        train_method = train

    elif task == "Weighted":
        if honeypot_pos is not None:
            warnings.warn("honeypot_pos is ignored")
        from WeightedVGG16 import WeightedVGG16, train
        model = WeightedVGG16(num_classes=args.num_classes, h_factor=h_factor)
        train_method = train

    elif "Native" in task:
        from NativeVGG16 import NativeVGG16
        from WeightedVGG16 import train
        model = NativeVGG16(num_classes=args.num_classes)
        train_method = train

    else:
        exit(1)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Trainable")

    return model, train_method


def load_datasets(args):
    batch_size = args.batch_size

    from datasets import CIFAR10Backdoor
    pre_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10Backdoor(args, train=True, pre_transform=pre_transform, post_transform=post_transform)
    test_dataset = CIFAR10Backdoor(args, train=False, pre_transform=pre_transform, post_transform=post_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

    # if args.dataset == "IMAGENET":
    #     train_dir = str(pathlib.Path(data_path) / "train")
    #     val_dir = str(pathlib.Path(data_path) / "val")
    #
    #     # Define transformations
    #     transform = transforms.Compose([
    #         transforms.Resize((224, 224)),  # Resize images to 224x224
    #         transforms.ToTensor(),  # Convert images to tensors
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         # Normalize using ImageNet stats
    #     ])
    #
    #     train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    #     val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def store_csv(losses, filename, path=None):
    if path is not None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filename = path / filename

    with open(f"{filename}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for step_loss in losses:
            writer.writerow([step_loss])

    print(f"Saved to {filename}.csv")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
