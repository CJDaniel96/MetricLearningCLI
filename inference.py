import argparse
import os
import torch
import pandas as pd
import numpy as np
import shutil
import cv2
import joblib
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from utils import setup_seed, DataTransformFactory, DataStatistics, load_model
from model import EfficientArcFaceModel, DOLGModel, MLGModel


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

def load_dataset(dataset_pkl):
    """
    Load a dataset from a pickle file.

    Args:
        dataset_pkl (str): The path to the dataset pickle file.

    Returns:
        dataset: The loaded dataset.
    """
    return joblib.load(dataset_pkl)

def create_inference_model(model_path, model_structure, embedding_size, faiss_index, threshold):
    """
    Create an inference model.

    Args:
        model_path (str): The path to the model weights.
        model_structure (str): The structure of the model.
        embedding_size (int): The size of the embedding layer.
        faiss_index (str): The path to the faiss index file.
        threshold (float): The threshold for matching.

    Returns:
        InferenceModel: The created inference model.
    """
    model = load_model(model_structure, model_path, embedding_size)
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.9)
    inference_model = InferenceModel(model, match_finder=match_finder)
    inference_model.load_knn_func(faiss_index)
    
    return inference_model

def knn_inference(model, data, train_dataset, image_type, mean, std, device, top, save_image_folder):
    result = []
    dataset = ImageFolder(train_dataset, DataTransformFactory.create_transform('train', mean, std))
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.9)
    inference_model = InferenceModel(model, match_finder=match_finder)
    inference_model.train_knn(dataset)

    if Path(data).is_file():
        nearest_imgs = []
        image_tensor = trans(data, DataTransformFactory.create_transform('train', mean, std), device)
        _, indices = inference_model.get_nearest_neighbors(image_tensor, top)
        for indice in indices.cpu()[0]:
            nearest_imgs.append(dataset[indice][0])
            result.append(dataset[indice][1])
        maxlabel = max(result, key=result.count)
        label = list(dataset.class_to_idx.keys())[maxlabel]
        
        print(f'Nearest Neighbor is {label}')
        
        if save_image_folder:
            Path(save_image_folder, label).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data, str(Path(save_image_folder).joinpath(label, Path(data).name)))
            if nearest_imgs:
                imsave(make_grid(nearest_imgs), mean, std, save_image_folder, 'nearest_imgs')
    elif Path(data).is_dir():
        for image_path in tqdm(Path(data).rglob(f'*.{image_type}')):
            nearest_imgs = []
            image_tensor = trans(str(image_path), DataTransformFactory.create_transform('train', mean, std), device)
            _, indices = inference_model.get_nearest_neighbors(image_tensor, top)
            for indice in indices.cpu()[0]:
                nearest_imgs.append(dataset[indice][0])
                result.append(dataset[indice][1])
            maxlabel = max(result, key=result.count)
            label = list(dataset.class_to_idx.keys())[maxlabel]
            
            if save_image_folder:
                dst = Path(save_image_folder, label)
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, str(dst.joinpath(Path(image_path).name)))
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

def general_inference(model, data, query_image, device, mean, std, top, image_type, image_size, inference_model, dataset_pkl, save_image_folder):
    result = []
    dataset = load_dataset(dataset_pkl)
    classes = dataset.classes

    if Path(data).is_file():
        query_features = extract_query_features(model, query_image, image_size, device, mean, std)
        output = extract_query_features(model, data, image_size, device, mean, std)
        is_match = inference_model.is_match(query_features, output)
        score = torch.cosine_similarity(output, query_features)
        result.append([score.detach().item(), data])
    elif Path(data).is_dir():
        for image in tqdm(sorted(Path(data).rglob(f'*.{image_type}'))):
            query_features = extract_query_features(model, query_image, image_size, device, mean, std)
            output = extract_query_features(model, str(image), image_size, device, mean, std)
            score = torch.cosine_similarity(output, query_features)
            result.append([score.detach().item(), str(image)])
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
    dataset = ImageFolder(train_dataset, DataTransformFactory.create_transform('train', mean, std))
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.9)
    inference_model = InferenceModel(model, match_finder=match_finder)
    inference_model.train_knn(dataset)

    result = []
    knn_result = []

    if Path(data).is_file():
        query_features = extract_query_features(model, query_image, image_size, device, mean, std)
        output = extract_query_features(model, data, image_size, device, mean, std)
        score = torch.cosine_similarity(output, query_features)
        
        nearest_imgs = []
        image_tensor = trans(data, DataTransformFactory.create_transform('train', mean, std), device)
        _, indices = inference_model.get_nearest_neighbors(image_tensor, top)
        for indice in indices.cpu()[0]:
            nearest_imgs.append(dataset[indice][0])
            knn_result.append(dataset[indice][1])
        maxlabel = max(knn_result, key=knn_result.count)
        label = list(dataset.class_to_idx.keys())[maxlabel]
        result.append([score.detach().item(), label, data])
        
        for i, (score, lb, im) in enumerate(result):
            print(f'Rank {i + 1} Score: {score}, Label: {lb}, Image Path: {im}')

    elif Path(data).is_dir():
        for image in tqdm(sorted(Path(data).rglob(f'*.{image_type}'))):
            query_features = extract_query_features(model, query_image, device, mean, std)
            output = extract_query_features(model, str(image), device, mean, std)
            score = torch.cosine_similarity(output, query_features)

            nearest_imgs = []
            image_tensor = trans(str(image), DataTransformFactory.create_transform('train', mean, std), device)
            _, indices = inference_model.get_nearest_neighbors(image_tensor, top)
            for indice in indices.cpu()[0]:
                nearest_imgs.append(dataset[indice][0])
                knn_result.append(dataset[indice][1])
            maxlabel = max(knn_result, key=knn_result.count)
            label = list(dataset.class_to_idx.keys())[maxlabel]
            result.append([score.detach().item(), label, str(image)])
    
    result.sort(reverse=True)
    result_df = pd.DataFrame(result, columns=['Score', 'Label', 'ImagePath'])

    if top > len(result):
        top = len(result)

    if save_image_folder:
        for i in range(top):
            if result[i][1] == query_label:
                shutil.copyfile(result[i][2], os.path.join(save_image_folder, os.path.basename(result[i][2])))

    result_df.to_csv(os.path.join(save_image_folder, 'result.csv'))

def main(opt):
    setup_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Use Device: {device}')

    if opt.save_image_folder and not os.path.exists(os.path.join(opt.save_image_folder)):
        os.makedirs(os.path.join(opt.save_image_folder))

    if opt.mean_std_file:
        mean, std = DataStatistics.get_mean_std(Path(opt.mean_std_file))

    if opt.model_structure == 'EfficientArcFaceModel':
        model = EfficientArcFaceModel(embedding_size=opt.embedding_size).to(device)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        model.eval()
    elif opt.model_structure == 'DOLGModel':
        model = DOLGModel(embedding_size=opt.embedding_size).to(device)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        model.eval()
    elif opt.model_structure == 'MLGModel':
        model = MLGModel(embedding_size=opt.embedding_size).to(device)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        model.eval()

    if opt.mode == 'total':
        total_inference(model, opt.train_dataset, opt.data, opt.query_image, opt.query_label, device, mean, std, opt.top, opt.image_type, opt.image_size, opt.save_image_folder)
    elif opt.mode == 'knn':
        knn_inference(model, opt.data, opt.train_dataset, opt.image_type, mean, std, device, opt.top, opt.save_image_folder)
    else:
        inference_model = create_inference_model(opt.weights, opt.model_structure, opt.embedding_size, opt.faiss_index, opt.threshold)
        general_inference(model, opt.data, opt.query_image, device, mean, std, opt.top, opt.image_type, opt.image_size, inference_model, opt.dataset_pkl, opt.save_image_folder)

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
    parser.add_argument('--model-structure', type=str, choices=['EfficientArcFaceModel', 'DOLGModel', 'MLGModel'], default='EfficientArcFaceModel')
    parser.add_argument('--embedding-size', type=int, default='512')
    parser.add_argument('--dataset-pkl', type=str, default='dataset.pkl')
    parser.add_argument('--faiss-index', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=5)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))