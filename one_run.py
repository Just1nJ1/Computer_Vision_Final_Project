import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from helper import load_model, load_datasets, set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Model name")
    parser.add_argument('--task', type=str, required=False, default="Honeypot",
                        help="[Honeypot, Weighted, Honeypot_Native, Weighted_Native]")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--batch_size', type=int, required=False, default=64, help="Batch size")
    parser.add_argument('--honeypot_pos', type=int, required=False, default=0, help="Honeypot position")
    parser.add_argument('--lamda', type=float, required=False, default=0.2, help="Lambda parameter")
    parser.add_argument('--warmup_steps', type=int, required=False, default=1000, help="Warmup steps")
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(2 ** 30), help="Random seed")
    parser.add_argument('--result_path', type=str, required=False, default="results", help="Results folder")
    parser.add_argument('--ckpt_path', type=str, required=False, default="ckpts", help="Checkpoints folder")
    parser.add_argument('--data_path', type=str, required=False, default="data", help="Data folder")
    parser.add_argument('--h_factor', type=int, required=False, default=7, help="h_factor")
    parser.add_argument('--lr', type=float, required=False, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', type=float, required=False, default=0.9, help="Momentum")
    parser.add_argument('--num_classes', type=int, required=False, default=10, help="Number of classes")
    parser.add_argument('--dataset', type=str, required=False, default="CIFAR10", help="Dataset name")
    parser.add_argument('--trigger_color', type=int, required=False, default=0, help="Trigger color")
    parser.add_argument('--target_label', type=int, required=False, default=0, help="Target label")
    parser.add_argument('--device', type=str, required=False, help="Device")
    parser.add_argument('--t', type=int, required=False, default=10, help="t")
    parser.add_argument('--threshold', type=float, required=False, default=0.8, help="Threshold")

    # parser.add_argument('--freeze', action='store_true', help="Freeze parameters")
    args = parser.parse_args()
    args.poison_rate = 0.05 if "Honeypot" in args.task else 0
    args.epochs = 2

    set_seed(args.seed)

    positions = [2, 3]
    base_name = args.name

    #### Honeypot
    for position in positions:
        try:
            args.honeypot_pos = position
            args.name = base_name + "_" + str(args.honeypot_pos)

            print("########### Parameters ###########")
            for arg, value in vars(args).items():
                print(f"{arg:<15}{str(value):<10}")

            print("########## Load Dataset ##########")
            train_loader, test_loader = load_datasets(args)

            print("########### Load Model ###########")
            model, train = load_model(args)

            # Parameters
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            device = torch.device("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else
                                  "cpu") if args.device is None else torch.device(args.device)

            print("########## Devices used ##########")
            print(device)

            args.model = model
            args.device = device
            args.train_loader = train_loader
            args.val_loader = test_loader
            args.optimizer = optimizer

            print("######### Start Training #########")
            train(args)

            print("########## Model Saving ##########")
            Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f'{args.ckpt_path}/{args.name}_{args.task}_final.pth')
        except:
            print(f"========== ERROR: {args.name} ==========")

    # args.num_classes = 100
    # args.dataset = 'CIFAR100'
    #
    # #### Honeypot CIFAR100
    # for position in positions:
    #     try:
    #         args.honeypot_pos = position
    #         args.name = base_name + "_CIFAR100_" + str(args.honeypot_pos)
    #
    #         print("########### Parameters ###########")
    #         for arg, value in vars(args).items():
    #             print(f"{arg:<15}{str(value):<10}")
    #
    #         print("########## Load Dataset ##########")
    #         train_loader, test_loader = load_datasets(args)
    #
    #         print("########### Load Model ###########")
    #         model, train = load_model(args)
    #
    #         # Parameters
    #         optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #         device = torch.device("cuda" if torch.cuda.is_available() else
    #                               "mps" if torch.backends.mps.is_available() else
    #                               "cpu") if args.device is None else torch.device(args.device)
    #
    #         print("########## Devices used ##########")
    #         print(device)
    #
    #         args.model = model
    #         args.device = device
    #         args.train_loader = train_loader
    #         args.val_loader = test_loader
    #         args.optimizer = optimizer
    #
    #         print("######### Start Training #########")
    #         train(args)
    #
    #         print("########## Model Saving ##########")
    #         Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
    #         torch.save(model.state_dict(), f'{args.ckpt_path}/{args.name}_{args.task}_final.pth')
    #     except:
    #         print(f"========== ERROR: {args.name} ==========")