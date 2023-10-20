import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from model import EfficientArcFaceModel, TransformerArcFaceModel
from utils import setup_seed, select_data_transforms, get_mean_std, save_mean_std, save_class_to_idx


def history_record(opt):
    save_dir = '\\\\?\\' + opt.save_dir if len(opt.save_dir) > 100 else opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_object = json.dumps(vars(opt))
    with open(os.path.join(save_dir, 'history.json'), 'w') as outfile:
        outfile.write(json_object)

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def train(model, epochs, train_loader, val_loader, train_dataset, val_dataset, device, optimizer, loss_optimizer, criterion, save_dir):
    best_loss = np.inf
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss_optimizer.zero_grad()
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            loss_optimizer.step()

            train_loss += loss.item()

        # validation
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                embeddings = model(inputs)
                loss = criterion(embeddings, labels)

                val_loss += loss.item()

        print(
            f'[{epoch + 1:03d}/{epochs:03d}]',
            f'Train Loss: {train_loss/len(train_loader):3.6f}',
            f'| Val Loss: {val_loss/len(val_loader):3.6f}'
        )
        print()

        torch.save(
            model,
            os.path.join(
                save_dir, f'Epoch_{epoch+1}_Loss_{val_loss/len(val_loader):.6f}.pt')
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, os.path.join(save_dir, 'best.pt'))
            print(f'saving best model with loss {val_loss/len(val_loader):.6f}')
            print()


def main(data_dir, epochs, batch_size, num_classes, embedding_size, lr, loss_lr, seed, early_stop_patience, save_dir):
    setup_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    data_dir = '\\\\?\\' + data_dir if len(data_dir) > 100 else data_dir
    save_dir = '\\\\?\\' + save_dir if len(save_dir) > 100 else save_dir

    mean, std = get_mean_std(data_dir, batch_size)
    save_mean_std(data_dir, mean, std)

    train_dataset = ImageFolder(os.path.join(
        data_dir, 'train'), select_data_transforms('train', mean, std))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = ImageFolder(os.path.join(
        data_dir, 'val'), select_data_transforms('train', mean, std))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f'class_to_idx: {train_dataset.class_to_idx}')
    save_class_to_idx(data_dir, train_dataset.class_to_idx)

    model = EfficientArcFaceModel(embedding_size=embedding_size).to(device)
    criterion = losses.SubCenterArcFaceLoss(
        num_classes=num_classes, embedding_size=embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_optimizer = torch.optim.Adam(criterion.parameters(), lr=loss_lr)
    train(model, epochs, train_loader, val_loader, train_dataset,
          val_dataset, device, optimizer, loss_optimizer, criterion, save_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss-lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early-stop-patience', type=int, default=3)
    parser.add_argument('--save-dir', default='')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()
    history_record(opt)
    main(**vars(opt))
