import pandas as pd
from os import path


def read_file(filename, index=None):
    '''
    Reads file into pandas df
    '''
    ext = path.split(filename)[-1].split('.')[-1]

    if ext == 'csv':
        return pd.read_csv(filename, index_col=index)
    elif ext == 'xls':
        return pd.read_excel(filename, index_col=index)
    elif ext == 'pkl':
        return pd.read_pickle(filename)
    else:
        print("Not a valid filetype")


def check_missing(df):
    '''
    Print column names,  number of missing rows
    for columns with missing values
    '''

    for col in df.columns:
        if df[col].isnull().any():
            num_missing = df[col].isnull().sum()
            print(col, num_missing)


def impute_missing(df, column, fill_type):
    '''
    Fill in missing values using method specified
    '''
    if fill_type == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    if fill_type == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)


def cap_extreme(df, column, lb=0.001, ub=0.999):
    '''
    cap extreme outliers using quantile specified as max value
    '''
    lb, ub = df[column].quantile(lb), df[column].quantile(ub)
    print('Monthly Income was capped between ${} and ${}.'.format(lb, ub))
    df[column] = df[column].apply(cap_value, args=(lb, ub))


def cap_value(x, lb, ub):
    '''
    helper function that returns cap for values exceeding it,
    0 for values
    itself otherwise
    '''
    if x > ub:
        return ub
    elif x < lb:
        return lb
    else:
        return x


def categorical_dummies(df, columns):
    '''
    CAN DO BEFORE SPLIT
    creates dummies for distinct categorical values including null
    drops original columns
    '''
    for column in columns:
        print(column)
        dummies = pd.get_dummies(df[column], prefix=column+"_is", prefix_sep='_', dummy_na=True)
        df = pd.concat([df, dummies], axis=1)

    df = df.drop(columns, axis=1)
    print(df.shape)

    return df

def join_df(df1, df2):
    return pd.concat([df1, df2], axis=1, join='inner')
