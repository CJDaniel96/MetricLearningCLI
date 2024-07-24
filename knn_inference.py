import argparse
import joblib
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from utils import read_mean_std, select_data_transforms, load_model, UnNormalize


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

def load_dataset(dataset_pkl):
    """
    Load a dataset from a pickle file.

    Args:
        dataset_pkl (str): The path to the dataset pickle file.

    Returns:
        dataset: The loaded dataset.
    """
    return joblib.load(dataset_pkl)

def process_image(image_path, mean, std):
    """
    Process an image.

    Args:
        image (PIL.Image): The image to process.
        mean (list): The mean values for the image.
        std (list): The standard deviation values for the image.

    Returns:
        Tensor: The processed image tensor.
    """
    image_array = cv2.imread(image_path)
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    data_transforms = select_data_transforms('train', mean, std)
    image_transforms = data_transforms(image)
    image_tensor = image_transforms.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return image_tensor

def main(opt):
    inference_model = create_inference_model(opt.model_path, opt.model_structure, opt.embedding_size, opt.faiss_index, opt.threshold)
    dataset = load_dataset(opt.dataset_pkl)
    classes = dataset.classes
    mean, std = read_mean_std(Path(opt.mean_std_file))
    unnormalize = UnNormalize(mean, std)
    
    if Path(opt.data).is_file():
        image_tensor = process_image(opt.data, mean, std)
        _, indices = inference_model.get_nearest_neighbors(image_tensor, opt.k)
        save_dir = Path(opt.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        rank = 0
        for indice in indices[0]:
            tensor, class_id = dataset[indice]
            class_name = classes[class_id]
            image = unnormalize(tensor)
            rank += 1
            save_image(image, save_dir / f'{class_name}_rank{rank}.jpg')

def run(**kwargs):
    opt = parse_opt(known=True)
    opt.__dict__.update(kwargs)
    main(opt)
    return opt

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='')
    parser.add_argument('--dataset-pkl', type=str, default='dataset.pkl')
    parser.add_argument('--faiss-index', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--mean-std-file', type=str, default='mean_std.txt')
    parser.add_argument('--model-structure', type=str, default='EfficientArcFaceModel')
    parser.add_argument('--model-path', type=str, default='model.pt')
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--save-dir', type=str, default='')
    return parser.parse_args() if known else parser.parse_known_args()[0]
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)