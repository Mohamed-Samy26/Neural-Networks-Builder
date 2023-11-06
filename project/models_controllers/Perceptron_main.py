from project.models import Perceptron as model
import pandas as pd
from project.helpers import preprocessing as pre


def train_perceptron(epochs: int,
                     learning_rate: float,
                     bias: float,
                     atrs= ["Area", "Perimeter"],
                     class1='BOMBAY',
                     class2='CALI'):

    # init model
    perceptron = model.Perceptron(epochs=epochs, learning_rate=learning_rate, bias=bias)

    # load data
    main_df = pd.read_excel("Dry_Bean_Dataset.xlsx")

    # preprocess
    main_df = pre.imputation(main_df, inplace=True)
    main_df = pre.standardize_columns(main_df, inplace=True)
    main_df = pre.normalize_columns(main_df, inplace=True)
    main_df = pre.signum_encode(main_df, class1=class1, class2=class2, column="Class")

    train, test = pre.train_test_split(main_df, test_size=0.4, random_state=78)

    # features split
    x_train = train[atrs]
    y_train = train["Class"]

    # before
    print(perceptron)

    # train
    perceptron.train(x_train, y_train)

    # after
    print(perceptron)

    x_test = test[atrs]
    y_test = test["Class"]

    print('Model acc: ')
    print(perceptron.accuracy(x_test, y_test))

    perceptron.plot_decision_boundary(
        x_test, y_test, feature_names=atrs, labels=[class1, class2]
    )

    perceptron.plot_confusion_matrix(x_test, y_test, labels=[class1, class2])


if __name__ == "__main__":
    train_perceptron(epochs=30, learning_rate=0.001, bias=0.02)
