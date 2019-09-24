import argparse
import dataloader
from model import IEMSAModel
import torch
import torch.nn as nn
import torch.optim as optim


def epoch(epoch_idx, is_train):
    model.train() if is_train else model.eval()
    loader = train_loader if is_train else val_loader
    for batch_idx, batch in enumerate(loader):
        batch = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch['response'])
        if is_train:
            loss.backward()
            optimizer.step()
        if is_train and (batch_idx % args.log_interval == 0):
            print(epoch_idx, loss)


def train():
    for epoch_idx in range(1, args.epochs + 1):
        epoch(epoch_idx, is_train=True)
        epoch(epoch_idx, is_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--project', type=str, default='iemsa')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader = dataloader.load_dataloader(args.data_dir, args.batch_size, is_train=True)
    val_loader = dataloader.load_dataloader(args.data_dir, args.batch_size, is_train=False)
    model = IEMSAModel(args)
    model = model.to(device)
    optimizer = optim.Adam(args.lr)
    criterion = nn.NLLLoss()

    train()
