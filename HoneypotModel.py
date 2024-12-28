import time

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from tqdm import tqdm

from helper import store_csv


# Honeypot
class VGG16Modified(torch.nn.Module):
    def __init__(self, num_classes=10, honeypot_pos=0):
        super(VGG16Modified, self).__init__()
        self.honeypot_pos = honeypot_pos

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Up to the first part
        # self.features_part1 = nn.Sequential(*list(vgg16.features.children())[0:5])
        # self.features_part2 = nn.Sequential(*list(vgg16.features.children())[5:10])
        # self.features_part3 = nn.Sequential(*list(vgg16.features.children())[10:17])
        # self.features_part4 = nn.Sequential(*list(vgg16.features.children())[17:24])
        # self.features_part5 = nn.Sequential(*list(vgg16.features.children())[24:])
        if honeypot_pos == 0:
            self._honeypot_split = 5
        elif honeypot_pos == 1:
            self._honeypot_split = 10
        elif honeypot_pos == 2:
            self._honeypot_split = 17
        elif honeypot_pos == 3:
            self._honeypot_split = 24
        elif honeypot_pos == 4:
            self._honeypot_split = 30
        self.features_part1 = nn.Sequential(*list(vgg16.features.children())[:self._honeypot_split])
        self.features_part2 = nn.Sequential(*list(vgg16.features.children())[self._honeypot_split:])

        self.honey_pot = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(self.features_part1[-3].out_channels * 7 * 7, num_classes)
        )

        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, num_classes)

        self.classifier = nn.Sequential(*list(vgg16.classifier.children()))

    def forward(self, x):
        x = self.features_part1(x)
        honeypot_output = self.honey_pot(x)
        x = self.features_part2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, honeypot_output

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.honey_pot.parameters():
            param.requires_grad = True

        for param in self.classifier[-1].parameters():
            param.requires_grad = True

        print("########## Model Freeze ##########")
        self.print_trainable_layers()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

        print("######### Model Unfreeze #########")
        self.print_trainable_layers()

    def print_trainable_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Trainable")


def custom_entropy(task_predictions, honeypot_predictions, targets, true_label, c=0.05, num_classes=10, prev_L_CETask_mean=None):
    targets = F.one_hot(targets, num_classes)

    task_predictions = torch.softmax(task_predictions, dim=1)
    honeypot_predictions = torch.softmax(honeypot_predictions, dim=1)
    
    epsilon = 1e-8
    L_CEHoneypot = (-targets * torch.log(task_predictions + epsilon)).mean(dim=1)
    L_CETask = (-targets * torch.log(honeypot_predictions + epsilon)).mean(dim=1)
    L_CETask_mean = torch.mean(L_CETask).clamp(min=epsilon)

    if prev_L_CETask_mean is not None:
        prev_L_CETask_mean[:-1] = prev_L_CETask_mean[1:].clone().detach()
        prev_L_CETask_mean[-1] = L_CETask_mean.detach()
        L_CETask_mean = torch.mean(prev_L_CETask_mean)
    
    # L_CEHoneypot = (-targets * torch.log(task_predictions)).mean(dim=1)
    # L_CETask = (-targets * torch.log(honeypot_predictions)).mean(dim=1)
    # L_CETask_mean = torch.mean(L_CETask)
    W_x = L_CEHoneypot / L_CETask_mean

    # L_WCE = torch.sigmoid(W_x - min(c, torch.mean(W_x).item())) * L_CETask
    L_WCE = torch.sigmoid(W_x - c) * L_CETask
    loss = L_WCE.sum()

    predicted_labels = task_predictions.argmax(dim=1)

    # Mask for poisoned data (where true_label != -1)
    poisoned_mask = (true_label != -1)

    # Total number of poisoned samples
    total_poisoned = poisoned_mask.sum().item()

    # Successful attacks: Poisoned samples where predicted label != true label
    successful_attacks = (predicted_labels[poisoned_mask] != true_label[poisoned_mask]).sum().item()

    return loss, total_poisoned, successful_attacks


def train(args):
    name = args.name
    model = args.model
    device = args.device
    train_loader = args.train_loader
    val_loader = args.val_loader
    optimizer = args.optimizer
    epochs = args.epochs
    path = args.result_path

    criterion = custom_entropy
    prev_L_CETask_mean = torch.zeros(args.t, requires_grad=False).to(device)
    model.to(device)
    model.freeze()

    train_losses = []
    train_accs = []
    train_ASRs = []
    val_accs = []
    val_ASRs = []

    lowest_asr = 1

    step_count = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        num_poisoned = 0
        num_successful_attacks = 0

        for images, labels, true_label in (pbar := tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
            images, labels, true_label = images.to(device), labels.to(device), true_label.to(device)
            real_labels = torch.where(true_label != -1, true_label, labels)

            task, honeypot = model(images)

            loss, total_poisoned, successful_attacks = criterion(task, honeypot, labels, true_label, num_classes=args.num_classes, prev_L_CETask_mean=prev_L_CETask_mean)
            
            if torch.isnan(loss):
                print("NaN loss detected!")
                break
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = task.argmax(axis=1).eq(real_labels).sum().item() / labels.size(0)

            train_losses.append(loss.item())
            train_accs.append(accuracy)

            num_poisoned += total_poisoned
            num_successful_attacks += successful_attacks

            step_count += 1

            if step_count == args.warmup_steps:
                model.unfreeze()

            # Update tqdm progress bar
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "ASR": f"{num_successful_attacks / num_poisoned:.4f}"
            })

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        throughput = len(train_loader.dataset) / epoch_time
        train_ASRs.append(num_successful_attacks / num_poisoned)

        model.eval()
        correct = 0
        num_poisoned = 0
        num_successful_attacks = 0
        total = 0
        with torch.no_grad():
            for images, labels, true_label in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
                images, labels, true_label = images.to(device), labels.to(device), true_label.to(device)
                real_labels = torch.where(true_label != -1, true_label, labels)
                
                task, honeypot = model(images)
                loss, total_poisoned, successful_attacks = criterion(task, honeypot, labels, true_label)

                correct += task.argmax(axis=1).eq(real_labels).sum().item()
                num_poisoned += total_poisoned
                num_successful_attacks += successful_attacks
                total += images.size(0)

        val_accuracy = correct / total
        val_ASR = num_successful_attacks / num_poisoned

        val_accs.append(val_accuracy)
        val_ASRs.append(val_ASR)

        if val_ASR < lowest_asr:
            lowest_asr = val_ASR
            torch.save(model.state_dict(), f'{args.ckpt_path}/{args.name}_{args.task}_best.pth')

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Throughput: {throughput:.2f} images/sec, "
            f"Validation Accuracy: {val_accuracy:.4f}, "
            f"Validation ASR: {val_ASR:.4f}"
        )

        print(f"Lowest ASR: {lowest_asr:.4f}")

        store_csv(train_losses, f"{name}_train_loss", path)
        store_csv(train_accs, f"{name}_train_accs", path)
        store_csv(train_ASRs, f"{name}_train_ASRs", path)
        store_csv(val_accs, f"{name}_val_accs", path)
        store_csv(val_ASRs, f"{name}_val_ASRs", path)

        train_losses = []
        train_accs = []
        train_ASRs = []
        val_accs = []
        val_ASRs = []

    # from datetime import datetime
    # from zoneinfo import ZoneInfo
    #
    # eastern = ZoneInfo("America/New_York")
    #
    # current_datetime = datetime.now(eastern)
    #
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M")
    #
    # store_csv(train_losses, f"{formatted_datetime} - {name}_Honeypot_train_loss", path)
    # store_csv(train_accs, f"{formatted_datetime} - {name}_Honeypot_train_accs", path)
    # store_csv(train_ASRs, f"{formatted_datetime} - {name}_Honeypot_train_ASRs", path)
    # store_csv(val_accs, f"{formatted_datetime} - {name}_Honeypot_val_accs", path)
    # store_csv(val_ASRs, f"{formatted_datetime} - {name}_Honeypot_val_ASRs", path)
