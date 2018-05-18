import numpy as np
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from features import *

PREDICTED = {'fully_funded'}

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
    print("Missing Values:")
    #missing = pd.DataFrame(columns=['column_nmae', 'data_type'])
    missing = []
    for col in df.columns:
        if df[col].isnull().any():
            num_missing = df[col].isnull().sum()
            print(col, num_missing, df[col].dtype)
            missing.append(col)
    return missing


def impute_missing(df, column, fill_type):
    '''
    Fill in missing values using method specified
    '''
    if fill_type == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    if fill_type == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    if fill_type == 'zero':
        df[column].fillna(0, inplace=True)


def cap_extreme(df, column, lb=0.001, ub=0.999):
    '''
    cap extreme outliers using quantile specified as max value
    '''
    lb, ub = df[column].quantile(lb), df[column].quantile(ub)
    print('Column was capped between {} and {}.'.format(lb, ub))
    df.loc[:, column] = df[column].apply(cap_value, args=(lb, ub))


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

def convert_bad_boolean(df, column, val='t'):
    '''
    converts boolean value in inconsistent format
    to 1 if true, 0 if false
    '''
    df.loc[:,column] = df[column].apply(lambda x: 1 if x == val else 0)


def categorical_dummies(df, columns):
    '''
    naive dummies from categorical vars
    '''
    for col in columns:
        print(col)
        dummies = pd.get_dummies(df[col], prefix=col+"_is", prefix_sep='_', dummy_na=True)
        df = pd.concat([df, dummies], axis=1)

    df = df.drop(columns, axis=1)
    print(df.shape)

    return df

def top_categories(df, col, top_k=-1):
    '''
    get top five categories + nan/others
    make top k or top percent?

    '''
    if top_k != -1:
        dummies = set(df[col].value_counts().head(top_k).index)
    else:
        dummies = set(df[col].value_counts().index)
    dummies = dummies.union(set([np.nan, 'others']))

    return dummies

def dummify(dummies, df, col):
    '''
    '''
    for val in dummies:
        col_name = '{}_is_{}'.format(col, str(val))
        if val != 'others':
            df.loc[:, col_name] = df[col].apply(lambda x: 1 if x == val else 0)
        else:
            df.loc[:, col_name] = df[col].apply(lambda x: 1 if x not in dummies else 0)
        #df = pd.concat([df, df[col_name]], axis=1)


def disaggregate_thyme(df, col, interval):

    col_name = '{}_{}'.format(col, interval)
    if interval == 'year':
        df.loc[:,col_name] = df[col].apply(lambda x: x.year)
    elif interval == 'month':
        df.loc[:,col_name] = df[col].apply(lambda x: x.month)


def pre_process(train, test):
    '''
    applies parallel preprocessing fcns to train and test dataframes
    '''
    features = set()
    for col in train.columns:
        if col in BINARY:
            print('Booleans converted: {}'.format(col))
            convert_bad_boolean(train, col, 't')
            convert_bad_boolean(test, col, 't')
            features.add(col)
        if col in CATEGORICAL:
            print('Dummified: {}'.format(col))
            dummies = top_categories(train, col, 5)
            x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
            #print(x)
            features = features.union(x)
            dummify(dummies, train, col)
            dummify(dummies, test, col)
        if col in IMPUTE_BY:
            features.add(col)
            print('Missing values imputed: {}'.format(col))
            impute_missing(train, col, IMPUTE_BY[col])
            impute_missing(test, col, IMPUTE_BY[col])
        if col in OTHERS and NORMALIZE == True:
            print('Normalizing: {}'.format(col))
            features.add(col)
            train.loc[:, col] = normalize(train[[col]], axis=0)
            test.loc[:, col] = normalize(test[[col]],axis=0)
        if col in THYME:
            features.add('{}_month'.format(col))
            print('Getting month of {}'.format(col))
            disaggregate_thyme(train, col, 'month')
            disaggregate_thyme(test, col, 'month')

    features = features - set(PREDICTED)

    return train, test, features


def join_df(df1, df2):
    return pd.concat([df1, df2], axis=1, join='inner')
