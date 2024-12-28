import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=False, help="Model name")
parser.add_argument('--task', type=str, required=False, default="Honeypot", help="[Honeypot, Weighted, Honeypot_Native, Weighted_Native]")
parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
parser.add_argument('--batch_size', type=int, required=False, default=64, help="Batch size")
parser.add_argument('--honeypot_pos', type=int, required=False, default=0, help="Honeypot position")
parser.add_argument('--lamda', type=float, required=False, default=0.2, help="Lambda parameter")
parser.add_argument('--warmup_steps', type=int, required=False, default=1000, help="Warmup steps")
parser.add_argument('--seed', type=int, required=False, default=np.random.randint(2**30), help="Random seed")
parser.add_argument('--log_path', type=str, required=False, default="logs", help="Logs folder")
parser.add_argument('--ckpt_path', type=str, required=False, default="ckpts", help="Checkpoints folder")
parser.add_argument('--h_factor', type=int, required=False, default=7, help="h_factor")
parser.add_argument('--lr', type=float, required=False, default=1e-3, help="Learning rate")
parser.add_argument('--momentum', type=float, required=False, default=0.9, help="Momentum")
parser.add_argument('--num_classes', type=int, required=False, default=10, help="Number of classes")
parser.add_argument('--dataset', type=str, required=False, default="CIFAR10", help="Dataset name")

# parser.add_argument('--freeze', action='store_true', help="Freeze parameters")
args = parser.parse_args()