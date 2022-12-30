import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import argparse
import pickle
from func import  dis
from tqdm import tqdm
from train import train_model


def main(file, save_file, model=None):
    print("Start")
    with open(file, 'r', encoding='utf-8') as file:
        data = [i.rstrip('\n') for i in file.readlines()]

    df = []
    print("Generate Dataset")
    for text in tqdm(data):
        file1, file2 = text.split()
        with open(file1, 'r', encoding='utf8') as file:
            text1 = [i.rstrip('\n') for i in file.readlines()]
        with open(file2, 'r', encoding='utf8') as file:
            text2 = [i.rstrip('\n') for i in file.readlines()]
        df.append([len(text1), len(text2), dis(text1, text2)])
    df = pd.DataFrame(df, columns=['len1', 'len2', 'distance'])
    print("The dataset was created successfully")

    if model is None:
        print("Train model")
        model = train_model(['files', 'plagiat1', 'plagiat2'], 'model.pkt')
    else:
        print("Load Model")
        with open(model, 'rb') as file:
            model = pickle.load(file)
    pred_proba = model.predict(df)
    print("Save Result")
    with open(save_file, 'w') as file:
        file.write('\n'.join([str(i[0])[:5] for i in pred_proba]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('save_file', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    main(args.file, args.save_file, args.model)