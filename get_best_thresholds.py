import argparse
import torch
from pathlib import Path
from utils import setup_seed, DataStatistics, create_dataloader, get_classes_threshold
from model import EfficientArcFaceModel, DOLGModel, MLGModel


def main(opt):
    setup_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Use Device: {device}')

    if opt.mean_std_file:
        mean, std = DataStatistics.get_mean_std(Path(opt.mean_std_file))

    if opt.model_structure == 'EfficientArcFaceModel':
        model = EfficientArcFaceModel(embedding_size=opt.embedding_size).to(device)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        model.eval()
    elif opt.model_structure == 'DOLG':
        model = DOLGModel(embedding_size=opt.embedding_size).to(device)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        model.eval()
    elif opt.model_structure == 'MLGModel':
        model = MLGModel(embedding_size=opt.embedding_size).to(device)
        model.load_state_dict(torch.load(opt.weights))
        model.cuda()
        model.eval()
        
    _, dataset = create_dataloader(
        opt.data_dir, 'train', opt.image_size,
        opt.batch_size, False, opt.num_workers,
        opt.seed, mean, std
    )
    
    df_thresholds, df_cross_stats = get_classes_threshold(dataset, model, Path(opt.save_dir))
    
    print("Within-class stats:\n", df_thresholds)
    print("Cross-class mean:\n", df_cross_stats["mean"])
    print("Cross-class max:\n", df_cross_stats["max"])
    print("Cross-class min:\n", df_cross_stats["min"])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--model-structure', type=str, choices=['EfficientArcFaceModel', 'DOLGModel', 'MLGModel'], default='EfficientArcFaceModel')
    parser.add_argument('--mean-std-file', type=str, default='', help='e.x. /path/to/mean_std.txt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-dir', default='')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)