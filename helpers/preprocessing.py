import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame, columns: list = None, inplace: bool = False):
    """Normalize the values of the given columns of a dataframe.
    make all values between 0 and 1
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to normalize.
    columns : list, optional
        The columns to normalize. If None, all columns are normalized.
        The default is None.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.

    Returns
    -------
    pd.DataFrame
        The normalized dataframe.
    """

    normalized_df = df

    if columns is None:
        columns = df.select_dtypes(include=["float64", "int64"]).columns

    if not inplace:
        normalized_df = df.copy()

    for column in columns:
        normalized_df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )

    return df


def imputation(
    df: pd.DataFrame, columns: list = None, inplace: bool = False, method: str = "mean"
):
    """Fill the NaN values of the given columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to fill.
    columns : list, optional
        The columns to fill. If None, all columns are filled.
        The default is None.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.

    method : str, optional
        The method to use for filling. Can be "mean", "median" or "mode".
        The default is "mean".

    Returns
    -------
    pd.DataFrame
        The filled dataframe.
    """

    filled_df = df

    if columns is None:
        columns = df.select_dtypes(include=["float64", "int64"]).columns

    if not inplace:
        filled_df = df.copy()

    for column in columns:
        if method == "mean":
            filled_df[column].fillna(df[column].mean(), inplace=True)
        elif method == "median":
            filled_df[column].fillna(df[column].median(), inplace=True)
        elif method == "mode":
            filled_df[column].fillna(df[column].mode(), inplace=True)
        else:
            raise ValueError(f"Invalid method: {method}")

    return filled_df


def one_hot_encoding(df: pd.DataFrame, columns: list = None, inplace: bool = False):
    """One hot encode the given columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to one hot encode.
    columns : list, optional
        The columns to one hot encode. If None, all columns are one hot encoded.
        The default is None.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.

    Returns
    -------
    pd.DataFrame
        The one hot encoded dataframe.
    """

    encoded_df = df

    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns

    if not inplace:
        encoded_df = df.copy()

    for column in columns:
        encoded_df = pd.concat(
            [encoded_df, pd.get_dummies(encoded_df[column], prefix=column)], axis=1
        )
        encoded_df.drop(column, axis=1, inplace=True)

    return encoded_df


def standardize_columns(df: pd.DataFrame, columns: list = None, inplace: bool = False):
    """Standardize the values of the given columns of a dataframe.
    make all values have mean 0 and std 1 to fit a normal distribution

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to standardize.
    columns : list, optional
        The columns to standardize. If None, all columns are standardized.
        The default is None.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.

    Returns
    -------
    pd.DataFrame
        The standardized dataframe.
    """

    standardized_df = df

    if columns is None:
        columns = df.select_dtypes(include=["float64", "int64"]).columns

    if not inplace:
        standardized_df = df.copy()

    for column in columns:
        standardized_df[column] = (df[column] - df[column].mean()) / df[column].std()

    return standardized_df


def signum_encode(
    df: pd.DataFrame, column: str, class1: str, class2: str, replace: bool = True
):
    """Signum encode the given columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to signum encode.
    column : str
        The column to signum encode.
    class1 : str
        The first class to encode.
    class2 : str
        The second class to encode.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.
    replace : bool, optional
        Whether to replace the original column. The default is False.

    Returns
    -------
    pd.DataFrame
        The signum encoded dataframe.
    """

    signum_encoded_df = df.copy()

    # filter out the rows that are not class1 or class2
    signum_encoded_df = signum_encoded_df[
        (signum_encoded_df[column] == class1) | (signum_encoded_df[column] == class2)
    ]
    if not replace:
        add_column = column + "_signum"
        signum_encoded_df[add_column] = signum_encoded_df[column]
        signum_encoded_df.loc[:, add_column] = (
            signum_encoded_df[add_column]
            .apply(lambda x: 1 if x == class1 else -1)
            .astype("float64")
        )
    else:
        signum_encoded_df.loc[:, column] = (
            signum_encoded_df[column]
            .apply(lambda x: 1 if x == class1 else -1)
            .astype("float64")
        )
    return signum_encoded_df


def label_encode(
    df: pd.DataFrame, column: str, inplace: bool = False, replace: bool = False
):
    """Label encode the given columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to label encode.
    column : str
        The column to label encode.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.
    replace : bool, optional
        Whether to replace the original column. The default is False.

    Returns
    -------
    pd.DataFrame
        The label encoded dataframe.
    """

    label_encoded_df = df
    map = dict(
        zip(
            label_encoded_df[column].astype("category").cat.codes,
            label_encoded_df[column],
        )
    )

    if not inplace:
        label_encoded_df = df.copy()

    if not replace:
        add_column = column + "_label"
        label_encoded_df[add_column] = (
            label_encoded_df[column].astype("category").cat.codes
        )
    else:
        label_encoded_df[column] = label_encoded_df[column].astype("category").cat.codes

    return label_encoded_df, map


def train_test_split(df: pd.DataFrame, test_size: float = 0.4, random_state: int = 78):
    """Split the dataframe into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    test_size : float, optional
        The size of the testing set. The default is 0.2.

    Returns
    -------
    pd.DataFrame
        The training set.
    pd.DataFrame
        The testing set.
    """

    train = df.sample(frac=1 - test_size, random_state=random_state)
    test = df.drop(train.index)

    return train, test


def split_by_class(
    df: pd.DataFrame,
    labels: list[str] = None,
    test_size: float = 0.4,
    random_state: int = 78,
    class_column: str = "Class",
):
    """Split the dataframe into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    test_size : float, optional
        The size of the testing set. The default is 0.4.
    random_state : int, optional
        The random state to use for reproducibility. The default is 78.
    class_column : str, optional
    labels : list[str]
        The labels to split by.

    Returns
    -------
    pd.DataFrame
        The training set.
    pd.DataFrame
        The testing set.
    """

    train = pd.DataFrame()
    test = pd.DataFrame()
    
    if labels is None:
        labels = df[class_column].unique()
    
    for label in labels:
        label_df = df[df[class_column] == label]
        label_train, label_test = train_test_split(
            label_df, test_size=test_size, random_state=random_state
        )
        train = pd.concat([train, label_train])
        test = pd.concat([test, label_test])

    return train, test


def xy_split(
    df: pd.DataFrame,
    y_column: str,
    test_size: float = 0.4,
    random_state: int = 78,
    x_columns: list[str] = None,
):
    """Split the dataframe into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    y_column : str
        The column to use as the y values.
    test_size : float, optional
        The size of the testing set. The default is 0.4.
    random_state : int, optional
        The random state to use for reproducibility. The default is 78.
    x_columns : list[str], optional
        The columns to use as the x values. If None, all columns except y_column are used.
        The default is None.

    Returns
    -------
    pd.DataFrame
        The training set.
    pd.DataFrame
        The testing set.
    """

    train, test = split_by_class(df, test_size=test_size, random_state=random_state)

    if x_columns is None:
        x_columns = df.columns.drop(y_column)

    train_x = train[x_columns]
    train_y = train[y_column]
    test_x = test[x_columns]
    test_y = test[y_column]

    return train_x, test_x, train_y, test_y


# # function to convert y values 0,1,2,... to one hot encoded list `[[1,0,0,...], [0,1,0,...], ...]`
# def y_to_one_hot(y: pd.Series):
#     """Convert a series of y values to one hot encoded list.

#     Parameters
#     ----------
#     y : pd.Series
#         The series to convert.

#     Returns
#     -------
#     list[list[float]]
#         The one hot encoded list.
#     """

#     values = [[1 if i == j else 0 for i in range(y.max() + 1)] for j in y]
#     # map of class names to one hot encoded values
#     one_hot_map = {
#         class_name: [1 if i == class_index else 0 for i in range(y.max() + 1)]
#         for class_index, class_name in enumerate(y.unique())
#     }

#     return np.array(values), one_hot_map

def y_to_one_hot(y: pd.Series, map:dict = None):
    """Convert a series of y values to one hot encoded list.

    Parameters
    ----------
    y : pd.Series
        The series to convert.

    Returns
    -------
    list[list[float]]
        The one hot encoded list.
    """

    if map is None:
        map = {
            class_name: [1 if i == class_index else 0 for i in range(y.max() + 1)]
            for class_index, class_name in enumerate(y.unique())
        }

    values = [map[class_name] for class_name in y]

    return np.array(values), map