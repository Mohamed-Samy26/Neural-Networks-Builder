
from project.models import adaline as ad
import pandas as pd
from project.helpers import preprocessing as pp


def main():
    adaline = ad.Adaline(epochs=30, learning_rate=0.0001, bias=0.02)
    main_df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    train, test = pp.train_test_split(main_df, test_size=0.4, random_state=78)

    train = pp.imputation(train, inplace=True)
    train = pp.standardize_columns(train, inplace=True)
    train = pp.normalize_columns(train, inplace=True)
    train = pp.signum_encode(train, class1="BOMBAY", class2="CALI", column="Class")

    X = train[["Area", "Perimeter"]]
    Y = train["Class"]

    print(adaline)
    adaline.train(X, Y)
    print(adaline)

    test = pp.imputation(test, inplace=True)
    test = pp.standardize_columns(test, inplace=True)
    test = pp.normalize_columns(test, inplace=True)
    test = pp.signum_encode(test, class1="BOMBAY", class2="CALI", column="Class")

    x_test = test[["Area", "Perimeter"]]
    y_test = test["Class"]

    print(adaline.accuracy(x_test, y_test))
    adaline.plot_decision_boundary(
        x_test, y_test, feature_names=["Area", "Perimeter"], labels=["BOMBAY", "CALI"]
    )
    adaline.plot_confusion_matrix(x_test, y_test, labels=["BOMBAY", "CALI"])


if __name__ == "__main__":
    main()
