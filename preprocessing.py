import pandas as pd

def normalize_columns(df: pd.DataFrame , columns: list = None, inplace: bool = False):
    """Normalize the values of the given columns of a dataframe.
    
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
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if not inplace:
        normalized_df = df.copy()
        
    for column in columns:
        normalized_df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        
    return df

def imputation(df: pd.DataFrame, columns: list = None,
                    inplace: bool = False, method: str = "mean"):
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
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
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
        columns = df.select_dtypes(include=['object']).columns
    
    if not inplace:
        encoded_df = df.copy()
        
    for column in columns:
        encoded_df = pd.concat([encoded_df, pd.get_dummies(encoded_df[column], prefix=column)], axis=1)
        encoded_df.drop(column, axis=1, inplace=True)
        
    return encoded_df

def standardize_columns(df: pd.DataFrame, columns: list = None, inplace: bool = False):
    """Standardize the values of the given columns of a dataframe.
    
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
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if not inplace:
        standardized_df = df.copy()
        
    for column in columns:
        standardized_df[column] = (df[column] - df[column].mean()) / df[column].std()
        
    return standardized_df

def preprocess(df: pd.DataFrame, columns: list = None, inplace: bool = False):
    """Preprocess the given columns of a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to preprocess.
    columns : list, optional
        The columns to preprocess. If None, all columns are preprocessed.
        The default is None.
    inplace : bool, optional
        Whether to modify the dataframe in place. The default is False.
    
    Returns
    -------
    pd.DataFrame
        The preprocessed dataframe.
    """
    
    preprocessed_df = df
    
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if not inplace:
        preprocessed_df = df.copy()
        
    preprocessed_df = normalize_columns(preprocessed_df, columns=columns, inplace=True)
    preprocessed_df = imputation(preprocessed_df, columns=columns, inplace=True)
    preprocessed_df = standardize_columns(preprocessed_df, columns=columns, inplace=True)
    
    return preprocessed_df

def signum_encode(df: pd.DataFrame, column:str, class1:str, class2:str,
                  inplace: bool = False, replace: bool = False):
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
    
    signum_encoded_df = df
    
    if not inplace:
        signum_encoded_df = df.copy()

    # filter out the rows that are not class1 or class2
    signum_encoded_df = signum_encoded_df[(signum_encoded_df[column] == class1) | (signum_encoded_df[column] == class2)] 
    if not replace:        
        add_column = column + "_signum"
        signum_encoded_df[add_column] = signum_encoded_df[column]
        signum_encoded_df[add_column] = signum_encoded_df[add_column].apply(lambda x: 1 if x == class1 else -1)
    else:
        signum_encoded_df[column] = signum_encoded_df[column].apply(lambda x: 1 if x == class1 else -1)
        
    return signum_encoded_df


    