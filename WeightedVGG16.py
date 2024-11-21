import time

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

# TODO: Add Validation accuracy, store model parameters, comparing with native VGG16

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

def train(name, model, device, train_loader, optimizer, epochs, verbose=False):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    model.freeze()

    if verbose:
        pass

    step_count = 0
    for epoch in range(epochs):
        start_time = time.time()

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for images, labels, _ in pbar:
                images, labels = images.to(device), labels.to(device)

                predictions = model(images)

                loss = criterion(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = torch.argmax(predictions, dim=1).eq(labels).sum().item() / labels.size(0)

                # Update metrics
                step_count += 1

                if step_count == 1000:
                    model.unfreeze()
                    if verbose:
                        pass

                # Update tqdm progress bar
                pbar.set_postfix({
                    "Loss": f"{loss:.4f}",
                    "Accuracy": f"{accuracy:.4f}",
                })

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        throughput = len(train_loader.dataset) / epoch_time  # Images per second

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Throughput: {throughput:.2f} images/sec"
        )
