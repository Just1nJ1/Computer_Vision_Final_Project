import torch.nn as nn
import torchvision.models as models


class NativeVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(NativeVGG16, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, num_classes)

        self.features = vgg16.features
        self.flatten = nn.Flatten()
        self.classifier = vgg16.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classifier[6].parameters():
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
