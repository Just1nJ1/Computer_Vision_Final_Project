import csv
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_model(task="Honeypot", honeypot_pos=None):
    if task == "Honeypot":
        assert honeypot_pos is not None, "Missing honeypot position"
        from HoneypotModel import VGG16Modified, train
        model = VGG16Modified(num_classes=10, honeypot_pos=honeypot_pos)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Trainable")

        return model, train
    elif task == "Weighted":
        if honeypot_pos is not None:
            warnings.warn("honeypot_pos is ignored")
        from WeightedVGG16 import WeightedVGG16, train
        model = WeightedVGG16(num_classes=10)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Trainable")

        return model, train

def load_datasets(poison_rate=0.05, lamda=0.2, batch_size=64, poison_train=True, poison_test=True):
    from datasets import CIFAR10Backdoor
    pre_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        pre_transform,
        post_transform
    ])

    if poison_train:
        train_dataset = CIFAR10Backdoor(root='./data', train=True, pre_transform=pre_transform,
                                        post_transform=post_transform, trigger_color=255, poison_rate=poison_rate,
                                        target_class=0, lamda=lamda)
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    if poison_test:
        test_dataset = CIFAR10Backdoor(root='./data', train=False, pre_transform=pre_transform,
                                       post_transform=post_transform, trigger_color=255, poison_rate=poison_rate,
                                       target_class=0, lamda=lamda)
    else:
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([pre_transform, post_transform]), download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def store_csv(losses, filename):
    with open(f"{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step_Loss"])  # Header
        for step_loss in losses:
            writer.writerow([step_loss])

    print(f"Step losses saved to {filename}.csv")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
