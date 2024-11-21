import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from helper import load_model, load_datasets, set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=False, default="Honeypot", help="[Honeypot, Weighted]")
    parser.add_argument('--name', type=str, required=True, help="Model name")
    parser.add_argument('--epochs', type=int, required=False, help="Epochs")
    parser.add_argument('--batch_size', type=int, required=False, help="Batch size")
    parser.add_argument('--honeypot_pos', type=int, required=False, help="Honeypot position")
    parser.add_argument('--lamda', type=float, required=False, help="Lambda parameter")
    parser.add_argument('--warmup_steps', type=int, required=False, help="Warmup steps")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    # parser.add_argument('--freeze', action='store_true', help="Freeze parameters")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(2**30)
    task = args.task
    name = args.name
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else 1000
    honeypot_pos = args.honeypot_pos if args.honeypot_pos is not None else 0
    num_epochs = args.epochs if args.epochs is not None else 10
    batch_size = args.batch_size if args.batch_size is not None else 64
    lamda = args.lamda if args.lamda is not None else 0.2
    poison_rate = 0.05 if task == "Honeypot" else 0
    lr = 0.001
    momentum = 0.9

    set_seed(seed)

    print("########### Parameters ###########")
    print(f"task: {task}\n"
          f"name: {name}\n"
          f"num_epochs: {num_epochs}\n"
          f"batch_size: {batch_size}\n"
          f"honeypot_pos: {honeypot_pos}\n"
          f"lamda: {lamda}\n"
          f"warmup_steps: {warmup_steps}\n"
          f"seed: {seed}\n")

    print("########## Load Dataset ##########")
    train_loader, test_loader = load_datasets(poison_rate=poison_rate, lamda=lamda, batch_size=batch_size)
    
    print("########### Load Model ###########")
    model, train = load_model(task=task, honeypot_pos=honeypot_pos)

    # Parameters
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")

    print("########## Devices used ##########")
    print(device)

    print("######### Start Training #########")
    train(name, model, device, train_loader, optimizer, num_epochs, verbose=True)

    print("########## Model Saving ##########")
    Path('./ckpts').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f'ckpts/{name}.pth')
