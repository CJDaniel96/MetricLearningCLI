import argparse
import os
import torch
import pandas as pd
import numpy as np
import shutil
import cv2
from PIL import Image
from pathlib import Path
from glob import glob
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder, FaissKMeans
from utils import setup_seed, read_mean_std, select_data_transforms
from model import EfficientArcFaceModel, DOLGModel


def imsave(img, mean, std, save_image_folder, title, figsize=(8, 4)):
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )
    img = inv_normalize(img)
    npimg = img.numpy()
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{save_image_folder}\\{title}.jpg')

def trans(image_path, data_transforms, device):
    image = cv2.imread(image_path)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transforms = data_transforms(image)
    image_tensor = image_transforms.unsqueeze(0).to(device)

    return image_tensor

def inference(model, image_path, data_transforms, device):
    image = Image.open(image_path).convert("RGB")
    image_transforms = data_transforms(image)
    image_tensor = image_transforms.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model(image_tensor)

    return embeddings

def knn_inference(model, data, train_dataset, image_type, mean, std, device, top, save_image_folder):
    result = []
    dataset = ImageFolder(train_dataset, select_data_transforms('train', mean, std))
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.9)
    inference_model = InferenceModel(model, match_finder=match_finder)
    inference_model.train_knn(dataset)

    if os.path.isfile(data):
        nearest_imgs = []
        image_tensor = trans(data, select_data_transforms('train', mean, std), device)
        distances, indices = inference_model.get_nearest_neighbors(image_tensor, top)
        for indice in indices.cpu()[0]:
            nearest_imgs.append(dataset[indice][0])
            result.append(dataset[indice][1])
        maxlabel = max(result, key=result.count)
        label = list(dataset.class_to_idx.keys())[maxlabel]
        
        if save_image_folder:
            if not os.path.exists(os.path.join(save_image_folder, label)):
                os.makedirs(os.path.join(save_image_folder, label))

            shutil.copyfile(image_path, os.path.join(save_image_folder, label, os.path.basename(image_path)))
            if nearest_imgs:
                imsave(make_grid(nearest_imgs), mean, std, save_image_folder, 'nearest_imgs')
    elif os.path.isdir(data):
        for image_path in tqdm(glob(os.path.join(data, '**', f'*.{image_type}'), recursive=True)):
            nearest_imgs = []
            image_tensor = trans(image_path, select_data_transforms('train', mean, std), device)
            distances, indices = inference_model.get_nearest_neighbors(image_tensor, top)
            for indice in indices.cpu()[0]:
                nearest_imgs.append(dataset[indice][0])
                result.append(dataset[indice][1])
            maxlabel = max(result, key=result.count)
            label = list(dataset.class_to_idx.keys())[maxlabel]
            
            if save_image_folder:
                if not os.path.exists(os.path.join(save_image_folder, label)):
                    os.makedirs(os.path.join(save_image_folder, label))

                shutil.copyfile(image_path, os.path.join(save_image_folder, label, os.path.basename(image_path)))
        if save_image_folder and nearest_imgs:
            imsave(make_grid(nearest_imgs), mean, std, save_image_folder, 'nearest_imgs')

def extract_query_features(model, image_path, image_size, device, mean, std):
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    image = cv2.imread(image_path)
    query = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    query_transform = data_transforms(query)
    query_tensor = query_transform.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features = model(query_tensor)
    
    return features

def general_inference(model, data, query_image, device, mean, std, top, image_type, image_size, save_image_folder):
    result = []

    if os.path.isfile(data):
        query_features = extract_query_features(model, query_image, image_size, device, mean, std)
        output = extract_query_features(model, data, image_size, device, mean, std)
        score = torch.cosine_similarity(output, query_features)
        result.append([score.detach().item(), data])
    elif os.path.isdir(data):
        for image in tqdm(glob(f'{data}\**\*.{image_type}', recursive=True)):
            query_features = extract_query_features(model, query_image, image_size, device, mean, std)
            output = extract_query_features(model, image, image_size, device, mean, std)
            score = torch.cosine_similarity(output, query_features)
            result.append([score.detach().item(), image])
    elif '*' in data:
        for image in tqdm(sorted(Path(data).parent.glob(Path(data).name))):
            query_features = extract_query_features(model, query_image, image_size, device, mean, std)
            output = extract_query_features(model, str(image), image_size, device, mean, std)
            score = torch.cosine_similarity(output, query_features)
            result.append([score.detach().item(), str(image)])
    
    result.sort(reverse=True)
    result_df = pd.DataFrame(result, columns=['Score', 'ImagePath'])

    if top > len(result):
        top = len(result)

    for i in range(top):
        if save_image_folder:
            shutil.copyfile(result[i][1], os.path.join(save_image_folder, os.path.basename(result[i][1])))
    
    result_df.to_csv(os.path.join(save_image_folder, 'result.csv'))

def total_inference(model, train_dataset, data, query_image, query_label, device, mean, std, top, image_type, image_size, save_image_folder):
    dataset = ImageFolder(train_dataset, select_data_transforms('train', mean, std))
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.9)
    inference_model = InferenceModel(model, match_finder=match_finder)
    inference_model.train_knn(dataset)

    result = []
    knn_result = []

    if os.path.isfile(data):
        query_features = extract_query_features(model, query_image, image_size, device, mean, std)
        output = extract_query_features(model, data, image_size, device, mean, std)
        score = torch.cosine_similarity(output, query_features)
        
        nearest_imgs = []
        image_tensor = trans(image, select_data_transforms('train', mean, std), device)
        distances, indices = inference_model.get_nearest_neighbors(image_tensor, top)
        for indice in indices.cpu()[0]:
            nearest_imgs.append(dataset[indice][0])
            knn_result.append(dataset[indice][1])
        maxlabel = max(knn_result, key=knn_result.count)
        label = list(dataset.class_to_idx.keys())[maxlabel]
        result.append([score.detach().item(), label, image])

    elif os.path.isdir(data):
        for image in tqdm(glob(f'{data}\**\*.{image_type}', recursive=True)):
            query_features = extract_query_features(model, query_image, device, mean, std)
            output = extract_query_features(model, image, device, mean, std)
            score = torch.cosine_similarity(output, query_features)

            nearest_imgs = []
            image_tensor = trans(image, select_data_transforms('train', mean, std), device)
            distances, indices = inference_model.get_nearest_neighbors(image_tensor, top)
            for indice in indices.cpu()[0]:
                nearest_imgs.append(dataset[indice][0])
                knn_result.append(dataset[indice][1])
            maxlabel = max(knn_result, key=knn_result.count)
            label = list(dataset.class_to_idx.keys())[maxlabel]
            result.append([score.detach().item(), label, image])
    
    result.sort(reverse=True)
    result_df = pd.DataFrame(result, columns=['Score', 'Label', 'ImagePath'])

    if top > len(result):
        top = len(result)

    if save_image_folder:
        for i in range(top):
            if result[i][1] == query_label:
                shutil.copyfile(result[i][2], os.path.join(save_image_folder, os.path.basename(result[i][2])))

    result_df.to_csv(os.path.join(save_image_folder, 'result.csv'))

def main(weights, data, train_dataset, query_image, query_label, image_type, image_size, seed, mean_std_file, top, confidence, save_image_folder, mode, model_structure, embedding_size):
    setup_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Use Device: {device}')

    if save_image_folder and not os.path.exists(os.path.join(save_image_folder)):
        os.makedirs(os.path.join(save_image_folder))

    if mean_std_file:
        mean, std = read_mean_std(mean_std_file)

    if model_structure == 'EfficientArcFaceModel':
        model = EfficientArcFaceModel(embedding_size=embedding_size).to(device)
        model.load_state_dict(torch.load(weights))
        model.cuda()
        model.eval()
    elif model_structure == 'DOLG':
        model = DOLGModel(embedding_size=embedding_size).to(device)
        model.load_state_dict(torch.load(weights))
        model.cuda()
        model.eval()

    if mode == 'total':
        total_inference(model, train_dataset, data, query_image, query_label, device, mean, std, top, image_type, image_size, save_image_folder)
    elif mode == 'knn':
        knn_inference(model, data, train_dataset, image_type, mean, std, device, top, save_image_folder)
    else:
        general_inference(model, data, query_image, device, mean, std, top, image_type, image_size, save_image_folder)

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--train-dataset', type=str, default='')
    parser.add_argument('--weights', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--confidence', type=float, default=0.0, help='give a confidence threshold to control inference result')
    parser.add_argument('--query-image', type=str, default='')
    parser.add_argument('--query-label', type=str, default='')
    parser.add_argument('--image-type', type=str, default='jpg')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--mean-std-file', type=str, default='', help='e.x. /path/to/mean_std.txt')
    parser.add_argument('--save-image-folder', type=str, default='', help='save inference image to folder by result')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--model-structure', type=str, default='EfficientArcFaceModel')
    parser.add_argument('--embedding-size', type=int, default='512')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))