
from models import Adaline as ad
import pandas as pd
from helpers import preprocessing as pp


def infer_adaline(feature1, feature2, y_col= "Class", labels=["BOMBAY", "CALI"],
                  epochs=30, learning_rate=0.02, bias=0.0,
                  use_bias=True, mse_threshold=0, df=None, test_size=0.4, random_state=78):
    
    if len(labels) != 2:
        raise ValueError("Labels must be of length 2")
    
    if df is None:
        df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    main_df = df.copy()
    adaline = ad.Adaline(epochs=epochs, learning_rate=learning_rate, bias=bias, use_bias=use_bias,
                         mse_threshold=mse_threshold)
    train, test = pp.split_by_class(main_df,labels, test_size=test_size, random_state=random_state)

    train = pp.imputation(train, inplace=True)
    train = pp.standardize_columns(train, inplace=True)
    train = pp.normalize_columns(train, inplace=True)
    train = pp.signum_encode(train, class1=labels[0], class2=labels[1], column=y_col)

    X = train[[feature1, feature2]]
    Y = train[y_col]

    print(adaline)
    adaline.train(X, Y)
    print(adaline)

    test = pp.imputation(test, inplace=True)
    test = pp.standardize_columns(test, inplace=True)
    test = pp.normalize_columns(test, inplace=True)
    test = pp.signum_encode(test, class1=labels[0], class2=labels[1], column=y_col)

    x_test = test[[feature1, feature2]]
    y_test = test[y_col]

    print(adaline.accuracy(x_test, y_test))
    adaline.plot_decision_boundary(
        x_test, y_test, feature_names=[feature1, feature2], labels=labels
    )
    adaline.plot_confusion_matrix(x_test, y_test, labels=labels)

