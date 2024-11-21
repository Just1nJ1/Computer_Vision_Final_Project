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


def custom_entropy(task_predictions, honeypot_predictions, targets, true_label, c=0.05):
    targets = F.one_hot(targets, 10)

    task_predictions = torch.softmax(task_predictions, dim=1)
    honeypot_predictions = torch.softmax(honeypot_predictions, dim=1)

    L_CEHoneypot = (-targets * torch.log(task_predictions)).mean(dim=1)
    L_CETask = (-targets * torch.log(honeypot_predictions)).mean(dim=1)
    L_CETask_mean = torch.mean(L_CETask)
    W_x = L_CEHoneypot / L_CETask_mean

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


def train(name, model, device, train_loader, optimizer, epochs, verbose=False):
    criterion = custom_entropy
    model.to(device)
    model.train()
    model.freeze()
    step_losses = []
    ASR = []

    if verbose:
        pass

    step_count = 0
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        num_poisoned = 0
        num_successful_attacks = 0
        total_images = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for images, labels, true_label in pbar:
                images, labels, true_label = images.to(device), labels.to(device), true_label.to(device)
                task, honeypot = model(images)

                loss, total_poisoned, successful_attacks = criterion(task, honeypot, labels, true_label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Store the step loss in the list
                step_losses.append(loss.item())

                # Update metrics
                correct += task.argmax(axis=1).eq(labels).sum().item()
                running_loss += loss.item() * images.size(0)
                total_images += images.size(0)
                num_poisoned += total_poisoned
                num_successful_attacks += successful_attacks
                step_count += 1

                if step_count == 1000:
                    model.unfreeze()
                    if verbose:
                        pass

                # Update tqdm progress bar
                pbar.set_postfix({
                    "Loss": f"{running_loss / total_images:.4f}",
                    "Acc": f"{correct / total_images:.4f}",
                    "ASR": f"{num_successful_attacks / num_poisoned:.4f}"
                })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / len(train_loader.dataset)
        epoch_time = time.time() - start_time
        throughput = len(train_loader.dataset) / epoch_time  # Images per second
        ASR.append(num_successful_attacks / num_poisoned)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}, "
            f"Throughput: {throughput:.2f} images/sec"
        )

    store_csv(step_losses, f"{name}_step_loss")
    store_csv(ASR, f"{name}_ASR")
