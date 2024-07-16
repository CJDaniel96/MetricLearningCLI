import argparse
import json
import os
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
import joblib
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from model import DOLGModel, EfficientArcFaceModel, MultiheadArcFaceModel
from utils import EarlyStopping, setup_seed, select_data_transforms, get_mean_std, save_mean_std, save_class_to_idx, read_mean_std


def history_record(opt):
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    json_object = json.dumps(vars(opt))
    with save_dir.joinpath('history.json').open('w') as outfile:
        outfile.write(json_object)

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def train(model, epochs, train_loader, val_loader, train_set, test_set, device, optimizer, loss_optimizer, scheduler, loss_scheduler, criterion, accuracy_calculator: AccuracyCalculator, num_classes, save_dir, early_stopping):
    writer = SummaryWriter(save_dir / 'logs')
    best_loss = np.inf
    best_knn_accuracy = 0.0
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
            scheduler.step()
            loss_scheduler.step()

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
        
        train_embeddings, train_labels = get_all_embeddings(train_set, model)
        test_embeddings, test_labels = get_all_embeddings(test_set, model)
        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
        print("Computing accuracy")
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, test_labels, train_embeddings, train_labels, False
        )
        breakpoint()
        print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
        print()
        
        knn = KNeighborsClassifier(n_neighbors=num_classes)
        knn.fit(train_embeddings, train_labels)

        test_predictions = knn.predict(test_embeddings)
        knn_accuracy = np.mean(test_predictions == test_labels)
        print(f'KNN accuracy: {knn_accuracy}')
        
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch + 1)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch + 1)
        writer.add_scalar('Accuracy/precision_at_1', accuracies["precision_at_1"], epoch + 1)
        writer.add_scalar('Accuracy/knn', knn_accuracy, epoch + 1)

        torch.save(
            model.state_dict(),
            str(save_dir.joinpath(f'Epoch_{epoch+1}_Loss_{train_loss/len(train_loader):.6f}.pt'))
        )
        
        knn_save_dir = save_dir.joinpath('knn')
        knn_save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(knn, knn_save_dir.joinpath(f'Epoch_{epoch+1}_Accuracy_{knn_accuracy:.6f}.pkl'))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_dir.joinpath('best.pt'))
            print(f'saving best model with loss {val_loss/len(val_loader):.6f}')
            print()
            
        if knn_accuracy > best_knn_accuracy:
            best_knn_accuracy = knn_accuracy
            joblib.dump(knn, knn_save_dir.joinpath('best_knn.pkl'))
            print(f'saving best knn model with accuracy {knn_accuracy}')
            print()
            
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break


def main(data_dir, epochs, batch_size, num_classes, image_size, embedding_size, pretrained_weights, lr, loss_lr, seed, 
         model_structure, loss_structure, optimizer_selection, early_stop_patience, save_dir):
    setup_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)

    if data_dir.joinpath('mean_std.txt').exists():
        mean, std = read_mean_std(data_dir.joinpath('mean_std.txt'))
    else:
        mean, std = get_mean_std(data_dir, batch_size)
        save_mean_std(data_dir, mean, std)

    train_dataset = ImageFolder(str(data_dir.joinpath('train')) , select_data_transforms('train', mean, std, image_size=image_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = ImageFolder(str(data_dir.joinpath('val')), select_data_transforms('train', mean, std, image_size=image_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f'class_to_idx: {train_dataset.class_to_idx}')
    save_class_to_idx(data_dir, train_dataset.class_to_idx)

    if model_structure == 'EfficientArcFaceModel':
        if pretrained_weights:
            model = EfficientArcFaceModel(embedding_size=embedding_size, pretrained=False).to(device)
            model.load_state_dict(torch.load(pretrained_weights))
            model.cuda()
        else:
            model = EfficientArcFaceModel(embedding_size=embedding_size).to(device)
    elif model_structure == 'DOLGModel':
        if pretrained_weights:
            model = DOLGModel(embedding_size=embedding_size, image_size=image_size, pretrained=False).to(device)
            model.load_state_dict(torch.load(pretrained_weights))
            model.cuda()
        else:
            model = DOLGModel(embedding_size=embedding_size, image_size=image_size).to(device)
    elif model_structure == 'MultiheadArcFaceModel':
        if pretrained_weights:
            model = MultiheadArcFaceModel(embedding_size=embedding_size, pretrained=False).to(device)
            model.load_state_dict(torch.load(pretrained_weights))
            model.cuda()
        else:
            model = MultiheadArcFaceModel(embedding_size=embedding_size).to(device)
    else:
        raise ValueError('model_structure not supported')

    if loss_structure == 'SubCenterArcFaceLoss':
        criterion = losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size).to(device)
    elif loss_structure == 'ArcFaceLoss':
        criterion = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size).to(device)
    else:
        raise ValueError('loss_structure not supported')
    
    if optimizer_selection == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_optimizer = torch.optim.Adam(criterion.parameters(), lr=loss_lr)
    elif optimizer_selection == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        loss_optimizer = torch.optim.SGD(criterion.parameters(), lr=loss_lr, momentum=0.9, weight_decay=1e-5)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    loss_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loss_optimizer, T_max=1000)
    
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    
    if early_stop_patience:
        early_stopping = EarlyStopping(patience=early_stop_patience)
    else:
        early_stopping = None
    
    train(model, epochs, train_loader, val_loader, train_dataset, val_dataset, device, optimizer, loss_optimizer, scheduler, loss_scheduler, criterion, accuracy_calculator, num_classes, save_dir, early_stopping)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained-weights', type=str, default='')
    parser.add_argument('--model-structure', type=str, default='EfficientArcFaceModel')
    parser.add_argument('--loss-structure', type=str, default='SubCenterArcFaceLoss')
    parser.add_argument('--optimizer-selection', type=str, default='Adam')
    parser.add_argument('--loss-lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)
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
