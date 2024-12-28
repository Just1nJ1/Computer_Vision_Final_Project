import time

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from helper import store_csv


class WeightedVGG16(nn.Module):
    def __init__(self, h_factor: int=7, num_classes=10):
        super(WeightedVGG16, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, num_classes)

        self.features = []

        self.features.append(nn.Sequential(*list(vgg16.features.children())[0:5]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[5:10]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[10:17]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[17:24]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[24:]))

        self.classifiers = []

        for feature in self.features:
            channel = feature[-3].out_channels
            h = w = h_factor

            self.classifiers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((h, w)),
                nn.Flatten(),
                nn.Linear(h * w * channel, num_classes),
            ))

        self.features = nn.ModuleList(self.features)

        self.classifiers = nn.ModuleList(self.classifiers)

        self.last_classifier = vgg16.classifier

        self.last_classifier.insert(0, nn.Flatten())

        self.weight_model = nn.Linear((len(self.classifiers) + 1) * num_classes, num_classes)

    def forward(self, x):
        fs = [x]
        clss = []
        for feature in self.features:
            fs.append(feature(fs[-1]))

        for i, classifier in enumerate(self.classifiers):
            clss.append(classifier(fs[i + 1]))

        clss.append(self.last_classifier(fs[-1]))
        clss = torch.cat(clss, dim=1)
        y = self.weight_model(clss)

        return y

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.classifiers:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.last_classifier.parameters():
            param.requires_grad = True

        for param in self.weight_model.parameters():
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


class WeightedVGG16_V2(nn.Module):
    def __init__(self, h_factor: int=7, num_classes=10):
        super(WeightedVGG16_V2, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, num_classes)

        self.features = []

        self.features.append(nn.Sequential(*list(vgg16.features.children())[0:5]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[5:10]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[10:17]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[17:24]))
        self.features.append(nn.Sequential(*list(vgg16.features.children())[24:]))

        self.classifiers = []

        for feature in self.features:
            channel = feature[-3].out_channels
            h = w = h_factor

            self.classifiers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((h, w)),
                nn.Flatten(),
                nn.Linear(h * w * channel, num_classes),
            ))

        self.features = nn.ModuleList(self.features)

        self.classifiers = nn.ModuleList(self.classifiers)

        self.last_classifier = vgg16.classifier

        self.last_classifier.insert(0, nn.Flatten())

        self.weight_model = nn.Linear((len(self.classifiers) + 1) * num_classes, num_classes)

    def forward(self, x):
        fs = [x]
        clss = []
        for feature in self.features:
            fs.append(feature(fs[-1]))

        for i, classifier in enumerate(self.classifiers):
            clss.append(classifier(fs[i + 1]))

        clss.append(self.last_classifier(fs[-1]))
        clss = torch.cat(clss, dim=1)
        y = self.weight_model(clss)

        return y

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.classifiers:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.last_classifier.parameters():
            param.requires_grad = True

        for param in self.weight_model.parameters():
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

def train(args):
    name = args.name
    model = args.model
    device = args.device
    train_loader = args.train_loader
    val_loader = args.val_loader
    optimizer = args.optimizer
    epochs = args.epochs
    path = args.result_path
    # honeypot = "Honeypot" in args.task
    honeypot = True

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.freeze()

    train_losses = []
    train_accs = []
    val_accs = []

    train_ASRs = []
    val_ASRs = []
    lowest_asr = 1
    best_acc = 0

    step_count = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        num_poisoned = 0
        num_successful_attacks = 0

        for images, labels, true_label in (pbar := tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
            images, labels, true_label = images.to(device), labels.to(device), true_label.to(device)
            real_labels = torch.where(true_label != -1, true_label, labels)

            predictions = model(images)

            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(predictions, dim=1)

            accuracy = predictions.eq(real_labels).sum().item() / labels.size(0)

            if honeypot:
                poisoned_mask = (true_label != -1)
                total_poisoned = poisoned_mask.sum().item()
                successful_attacks = (predictions[poisoned_mask] != true_label[poisoned_mask]).sum().item()
                num_poisoned += total_poisoned
                num_successful_attacks += successful_attacks
                asr = (num_successful_attacks / num_poisoned) if num_poisoned > 0 else 0
                train_ASRs.append(asr)

            train_losses.append(loss.item())
            train_accs.append(accuracy)

            step_count += 1

            if step_count == args.warmup_steps:
                model.unfreeze()

            # Update tqdm progress bar
            pbar.set_postfix({
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "ASR": f"{asr:.4f}" if honeypot else "",
            })

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        throughput = len(train_loader.dataset) / epoch_time  # Images per second

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, true_label in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
                images, labels, true_label = images.to(device), labels.to(device), true_label.to(device)
                real_labels = torch.where(true_label != -1, true_label, labels)

                predictions = model(images)
                predictions = torch.argmax(predictions, dim=1)
                correct += predictions.eq(real_labels).sum().item()
                total += labels.size(0)

                if honeypot:
                    poisoned_mask = (true_label != -1)
                    total_poisoned = poisoned_mask.sum().item()
                    successful_attacks = (predictions[poisoned_mask] == labels[poisoned_mask]).sum().item()
                    num_poisoned += total_poisoned
                    num_successful_attacks += successful_attacks

        val_accuracy = correct / total
        val_accs.append(val_accuracy)
        if honeypot:
            val_ASR = num_successful_attacks / num_poisoned
            val_ASRs.append(val_ASR)
            if val_ASR < lowest_asr:
                lowest_asr = val_ASR
                torch.save(model.state_dict(), f'{args.ckpt_path}/{args.name}_{args.task}_best.pth')
        else:
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save(model.state_dict(), f'{args.ckpt_path}/{args.name}_{args.task}_best.pth')

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Throughput: {throughput:.2f} images/sec, "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        if honeypot:
            store_csv(train_losses, f"{name}_Honeypot_train_loss", path)
            store_csv(train_accs, f"{name}_Honeypot_train_accs", path)
            store_csv(train_ASRs, f"{name}_Honeypot_train_ASRs", path)
            store_csv(val_accs, f"{name}_Honeypot_val_accs", path)
            store_csv(val_ASRs, f"{name}_Honeypot_val_ASRs", path)

            train_losses = []
            train_accs = []
            train_ASRs = []
            val_accs = []
            val_ASRs = []
        else:
            store_csv(train_losses, f"{name}_train_loss", path)
            store_csv(train_accs, f"{name}_train_acc", path)
            store_csv(val_accs, f"{name}_val_acc", path)

    # from datetime import datetime
    # from zoneinfo import ZoneInfo
    #
    # eastern = ZoneInfo("America/New_York")
    #
    # current_datetime = datetime.now(eastern)
    #
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M")
    #
    # if honeypot:
    #     store_csv(train_losses, f"{formatted_datetime} - {name}_Honeypot_train_loss", path)
    #     store_csv(train_accs, f"{formatted_datetime} - {name}_Honeypot_train_accs", path)
    #     store_csv(train_ASRs, f"{formatted_datetime} - {name}_Honeypot_train_ASRs", path)
    #     store_csv(val_accs, f"{formatted_datetime} - {name}_Honeypot_val_accs", path)
    #     store_csv(val_ASRs, f"{formatted_datetime} - {name}_Honeypot_val_ASRs", path)
    # else:
    #     store_csv(train_losses, f"{formatted_datetime} - {name}_train_loss", path)
    #     store_csv(train_accs, f"{formatted_datetime} - {name}_train_acc", path)
    #     store_csv(val_accs, f"{formatted_datetime} - {name}_val_acc", path)
