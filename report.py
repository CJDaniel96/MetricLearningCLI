import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def loss_vision(loss):
    model_record_list = os.listdir(loss)
    loss_record = []
    for model_record in model_record_list:
        if 'Epoch' in model_record:
            loss_record.append([int(model_record.split('_')[1]), float(model_record.split('_')[-1][:-3])])
    loss_record.sort()
    plt.plot(np.array(loss_record)[:, 1], label='Loss')
    plt.title('Loss Record')
    plt.legend()
    plt.savefig(os.path.join(loss, 'loss_record.jpg'))

def csv_hist_vision(csv, csv_target_label):
    result_df = pd.read_csv(csv)
    ok_score = []
    ng_score = []
    for i in range(len(result_df)):
        if f'\\{csv_target_label}\\' in result_df.loc[i, 'ImagePath']:
            ok_score.append(result_df.loc[i, 'Score'])
        else:
            ng_score.append(result_df.loc[i, 'Score'])

    plt.hist([ng_score, ok_score], label = ['NG Similarity Score', 'OK Similarity Score'], bins=50, stacked=True)
    plt.title('Query Image is OK')
    plt.xlabel('Similarity Score')
    plt.ylabel('Images Stack')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(csv), 'result_hist.jpg'))

def csv_scatter_vision(csv):
    plt.clf()
    result_df = pd.read_csv(csv)
    classes_dict = {}
    for i in range(len(result_df)):
        classes_name = os.path.basename(os.path.dirname(result_df.loc[i, 'ImagePath']))
        if classes_name in classes_dict:
            classes_dict[classes_name].append(result_df.loc[i, 'Score'])
        else:
            classes_dict[classes_name] = [result_df.loc[i, 'Score']]

    count = 1
    for key in classes_dict:
        color = 'r' if count == 1 else 'g' if count == 2 else 'b' if count == 3 else 'y'
        plt.scatter(x=[count] * len(classes_dict[key]), y=classes_dict[key], c=color, label=key)
        count += 1
    plt.ylim([-1, 2])
    plt.title('Data Scatter')
    plt.xlabel('Labels')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(csv), 'result_scatter.jpg'))

def main(csv, loss, csv_target_label):
    if csv:
        csv_hist_vision(csv, csv_target_label)
        csv_scatter_vision(csv)
    elif loss:
        loss_vision(loss)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='')
    parser.add_argument('--csv-target-label', type=str, default='OK')
    parser.add_argument('--loss', type=str, default='')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))