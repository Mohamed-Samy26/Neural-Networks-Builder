import numpy as np
import pandas as pd
from models.MultiLayerPrecepetron import MultiLayerPrecepetron as mlp
import models.LayerInfo as li
import helpers.preprocessing as pp
import pickle as pl

def infer_mlp(layers: list[li.LayerInfo]):
    df = pd.read_excel("./Dry_Bean_Dataset.xlsx")
    df = pp.label_encode(df, "Class", inplace=True, replace=True)
    df = pp.imputation(df)
    df = pp.normalize_columns(df)
    df = pp.standardize_columns(df)
    train_x, test_x, train_y, test_y = pp.xy_split(df, "Class")

    enc_train_y, map = pp.y_to_one_hot(train_y)
    enc_test_y = pp.y_to_one_hot(test_y, map)

    enc_train_x = np.array(train_x)[:, np.newaxis, :]
    enc_test_x = np.array(test_x)

    model = mlp(5, 3, layers, "sigmoid")

    model.train(enc_train_x, enc_train_y, 20, 0.1)
    acc = model.accuracy(enc_test_x, enc_test_y)

    print(acc)
    print(model.confusion_matrix(enc_test_x, enc_test_y))
    # pl.dump(model, open("model.pkl", "wb"))
