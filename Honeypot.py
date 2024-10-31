import csv
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm


# Backdoor dataset
class CIFAR10Backdoor(Dataset):
    def __init__(self, root, train=True, pre_transform=None, post_transform=None, trigger_color=0, poison_rate=0.05,
                 target_class=0, lamda=0.2):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.trigger_color = trigger_color
        self.poisoned_indices = np.random.choice(len(self.dataset), int(len(self.dataset) * poison_rate), replace=False)
        self.lamda = lamda

    def add_trigger(self, img):
        img[221:224, 221:224] = img[221:224, 221:224] * (1 - self.lamda) + self.trigger_color * self.lamda
        img = img.astype(np.uint8)
        return img

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.pre_transform:
            img = self.pre_transform(img)
        img = np.array(img)

        # Add trigger if this is a poisoned sample
        if index in self.poisoned_indices:
            img = self.add_trigger(img)
            label = self.target_class  # Change label to target class

        img = Image.fromarray(img)
        if self.post_transform:
            img = self.post_transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)


# Modified VGG16
class VGG16Modified(nn.Module):
    def __init__(self, num_classes=10, honeypot_location=5, freeze=True):
        super(VGG16Modified, self).__init__()

        # Load pretrained VGG-16 model and freeze parameters
        vgg16 = models.vgg16(pretrained=True)
        if freeze:
            for param in vgg16.parameters():
                param.requires_grad = False

        # Up to the first part
        self.features_part1 = nn.Sequential(*list(vgg16.features.children())[:honeypot_location])

        # Add a convolutional branch parallel to continue from the first pooling layer
        self.honey_pot = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, num_classes)
        )

        # Remaining VGG features (after the first part)
        self.features_part2 = nn.Sequential(*list(vgg16.features.children())[honeypot_location:])

        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, num_classes)

        self.classifier = nn.Sequential(*list(vgg16.classifier.children()))

    def forward(self, x):
        x = self.features_part1(x)
        honey_pot = self.honey_pot(x)
        x = self.features_part2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        main_prob = F.softmax(x, dim=1)
        honey_pot_prob = F.softmax(honey_pot, dim=1)
        return main_prob - honey_pot_prob


def load_datasets(poison_rate=0.05, lamda=0.2):
    pre_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10Backdoor(root='./data', train=True, pre_transform=pre_transform, post_transform=post_transform, trigger_color=255, poison_rate=poison_rate, target_class=0, lamda=lamda)
    # train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([pre_transform, post_transform]), download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def load_model(freeze=True):
    model = VGG16Modified(num_classes=10, honeypot_location=5, freeze=freeze)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Trainable")

    return model


def store_csv(losses, filename):
    with open(f"{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step_Loss"])  # Header
        for step_loss in losses:
            writer.writerow([step_loss])

    print(f"Step losses saved to {filename}.csv")


def train(model, device, train_loader, criterion, optimizer, epochs):
    model.to(device)
    model.train()
    step_losses = []

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total_images = 0
        step_count = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                # Store the step loss in the list
                step_losses.append(loss.item())

                # Update metrics
                correct += outputs.argmax(axis=1).eq(labels).sum().item()
                running_loss += loss.item() * images.size(0)
                total_images += images.size(0)
                step_count += 1

                # # Display current loss and accuracy every 100 steps
                # if step_count % 100 == 0:
                #     current_loss = running_loss / total_images
                #     current_accuracy = correct / total_images
                #     print(f"Step [{step_count}] - Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.4f}")

                # Update tqdm progress bar
                pbar.set_postfix({
                    "Loss": f"{running_loss / total_images:.4f}",
                    "Acc": f"{correct / total_images:.4f}"
                })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / len(train_loader.dataset)
        epoch_time = time.time() - start_time
        throughput = len(train_loader.dataset) / epoch_time  # Images per second

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}, "
            f"Throughput: {throughput:.2f} images/sec"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Model name")
    parser.add_argument('--epochs', type=int, required=False, help="Epochs")
    parser.add_argument('--freeze', action='store_true', help="Freeze parameters")
    args = parser.parse_args()

    name = args.name
    freeze = args.freeze

    num_epochs = args.epochs if args.epochs is not None else 10

    print("########## Load Dataset ##########")
    train_loader, test_loader = load_datasets()
    
    print("########### Load Model ###########")
    model = load_model(freeze=freeze)

    # Parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")

    print("########## Devices used ##########")
    print(device)

    print("######### Start Training #########")
    train(model, device, train_loader, criterion, optimizer, num_epochs)

    print("########## Model Saving ##########")
    torch.save(model.state_dict(), f'{name}.pth')
