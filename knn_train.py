import argparse
import joblib
from pathlib import Path
from utils import DataStatistics, DataTransformFactory, load_model
from torchvision.datasets import ImageFolder
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder


def main(opt):
    """
    Train a k-nearest neighbors (KNN) model using the provided options.

    Args:
        opt (object): An object containing the options for training the KNN model.

    Raises:
        ValueError: If the dataset is not found.

    Returns:
        None
    """
    model = load_model(opt.model_structure, opt.model_path, opt.embedding_size)
    mean, std = DataStatistics.get_mean_std(Path(opt.dataset_folder))
    if Path(opt.dataset_pkl).exists():
        dataset = joblib.load(opt.dataset_pkl)
    elif Path(opt.dataset_folder).exists():
        dataset_folder = Path(f'\\\\?\\{opt.dataset_folder}')
        dataset = ImageFolder(dataset_folder, DataTransformFactory.create_transform('train', mean, std))
    else:
        raise ValueError('Dataset not found')
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=opt.threshold)
    knn = InferenceModel(model, match_finder=match_finder)
    knn.train_knn(dataset)
    
    if Path(opt.save_path).suffix == '.index':
        knn.save_knn_func(opt.save_path)
        dataset_pkl_path = Path(opt.save_path).parent / 'dataset.pkl'
        joblib.dump(dataset, str(dataset_pkl_path))
    else:
        Path(opt.save_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(opt.save_path) / 'knn_func.index'
        dataset_pkl_path = Path(opt.save_path) / 'dataset.pkl'
        knn.save_knn_func(str(save_path))
        joblib.dump(dataset, str(dataset_pkl_path))
        

def run(**kwargs):
    """
    This function runs the main training process.

    Args:
        **kwargs: Additional keyword arguments that can be passed to the function.

    Returns:
        opt: The updated options object.

    """
    opt = parse_opt(known=True)
    opt.__dict__.update(kwargs)
    main(opt)
    return opt


def parse_opt(known=False):
    """
    Parse the command line arguments for the script.

    Args:
        known (bool): Flag indicating whether all arguments are known or not.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, default='data')
    parser.add_argument('--dataset-pkl', type=str, default='dataset.pkl')
    parser.add_argument('--model-path', type=str, default='model')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--model-structure', type=str, default='EfficientArcFaceModel')
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--save-path', type=str, default='knn_func.index')
    return parser.parse_args() if known else parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
