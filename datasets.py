import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


# Backdoor dataset
class CIFAR10Backdoor(Dataset):
    def __init__(self, args, train=True, pre_transform = None, post_transform = None):
        root = args.data_path
        trigger_color = args.trigger_color
        poison_rate = args.poison_rate
        target_class = args.target_label
        lamda = args.lamda
        assert args.dataset in ['CIFAR10', 'CIFAR100']
        if args.dataset == 'CIFAR10':
            self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        elif args.dataset == 'CIFAR100':
            self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
        else:
            assert False, 'Unsupported dataset'
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
        true_label = -1
        if self.pre_transform:
            img = self.pre_transform(img)
        img = np.array(img)

        # Add trigger if this is a poisoned sample
        if index in self.poisoned_indices:
            img = self.add_trigger(img)
            true_label = label
            label = self.target_class  # Change label to target class

        img = Image.fromarray(img)
        if self.post_transform:
            img = self.post_transform(img)

        return img, label, true_label

    def __len__(self):
        return len(self.dataset)
