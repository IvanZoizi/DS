import argparse

from sklearn.model_selection import StratifiedKFold, train_test_split
from create_inputtxt import InputTrain
import pandas as pd
import numpy as np
from func import  dis
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

FILE_NAME = 'train.txt'


def train_model(dirs, model_dir):
    try:
        train = InputTrain(FILE_NAME, dirs)
        train.get_list_shuffle()
        result = train.write_result()

        with open(FILE_NAME, 'r', encoding='utf8') as file:
            text = [i.rstrip('\n') for i in file.readlines()]

        data = []
        distances = []
        print("Start create Train dataset")
        for iter_name in tqdm(text):
            file1, file2 = iter_name.split()
            with open(file1, 'r', encoding='utf8') as f:
                text1 = [i.rstrip('\n') for i in f.readlines()]
            with open(file2, 'r', encoding='utf8') as f:
                text2 = [i.rstrip('\n') for i in f.readlines()]
            data.append([len(text1), len(text2), dis(text1, text2)])
            distances.append(dis(text1, text2))

        for i in data:
            i.append(i[2] / np.mean(distances) if i[2] / np.mean(distances) <= 1 else 1)

        df = pd.DataFrame(data, columns=['len1', 'len2', 'distance', 'target'])
        print(df.head(5))
        df[['len1', 'len2', 'distance', 'target']].to_csv("result.csv", index=None)
        print("The dataset was created successfully")

        print("Start create Model")

        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
        # model = LogisticRegressionCV(cv=skf, random_state=42, verbose=1, n_jobs=1)
        model = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(df[["len1", "len2", "distance"]], df[['target']], test_size=0.2,
                                                            random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"MSE - {mean_squared_error(y_test, pred)}")
        print("Save model")

        with open(model_dir, 'wb') as file_model:
            pickle.dump(model, file_model)

        return model
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=str, nargs='+')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    train_model(args.dirs, args.model)
