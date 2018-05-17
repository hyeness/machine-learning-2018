import numpy as np
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

'''
class Pipeline:
    def __init__():
        self.df = None
        self.features
        self.predicted
'''

# DATA CLEANING AGENDA

class Features:
    def __init__(self, filename, binary, categorical, continuous,
                geographical, id, pred, features):
        self.df = self.read_file(filename)
        self.binary = binary
        self.categorical = categorical
        self.numeric = continuous
        self.geography = geographical
        self.id = id
        self.missing = []

    def read_file(self, filename, index=None):
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

    def check_missing(self):
        '''
        Print column names,  number of missing rows
        for columns with missing values
        '''
        print("Missing Values:")
        for col in df.columns:
            if self.df[col].isnull().any():
                num_missing = df[col].isnull().sum()
                print(col, num_missing, self.df[col].dtype)
                self.missing.append(col)


# convert bad booleans from t,f to 1,0
BINARY =['fully_funded', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
         'school_kipp', 'school_charter_ready_promise', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
         'eligible_double_your_impact_match', 'eligible_almost_home_match']

# dummify categorical variables using top 5 most frequent values, other or missing otherwise
CATEGORICAL = ['school_metro', 'primary_focus_subject', 'primary_focus_area',
           'secondary_focus_subject', 'secondary_focus_area',
           'resource_type', 'grade_level', 'school_state', 'school_zip',
           'teacher_prefix']

# ignore for now
GEOGRAPHICAL = ['school_latitude', 'school_longitude', 'school_city', 'school_state',
                'school_county', 'school_state', 'school_zip', 'school_district']

# drop later
ID = ['teacher_acctid', 'schoolid', 'school_ncesid']


OTHERS = ['fulfillment_labor_materials', 'total_price_excluding_optional_support',
          'total_price_including_optional_support', 'students_reached']

DATE = ['date_posted']

PREDICTED = ['fully_funded']

IMPUTE_BY = {'students_reached': 'mean'}

NORMALIZE = True


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

def convert_bad_boolean(df, col, val='t'):
    '''
    converts boolean value in inconsistent format
    to 1 if true, 0 if false
    '''
    df[col] = df[col].apply(lambda x: 1 if x == val else 0)


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

def top_five_categories(df, col):
    '''
    get top five categories + nan/others
    '''
    dummies = set(df[col].value_counts().head().index)
    dummies = dummies.union(set([np.nan, 'others']))
    return dummies

def dummify(dummies, df, col):
    for val in dummies:
        col_name = '{}_is_{}'.format(col, str(val))
        if val != 'others':
            df[col_name] = df[col].apply(lambda x: 1 if x == val else 0)
        else:
            df[col_name] = df[col].apply(lambda x: 1 if x not in dummies else 0)
        df = pd.concat([df, df[col_name]], axis=1)


def pre_process(train, test):
    '''
    applies parallel preprocessing fcns to train and test dataframes
    '''
    features = set()
    for col in train.columns:
        if col in BINARY:
            #print('Booleans converted: {}'.format(col))
            convert_bad_boolean(train, col, 't')
            convert_bad_boolean(test, col, 't')
            features.add(col)
        if col in CATEGORICAL:
            #print('Dummified: {}'.format(col))
            dummies = top_five_categories(train, col)
            x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
            #print(x)
            features = features.union(x)
            dummify(dummies, train, col)
            dummify(dummies, test, col)
        if col in IMPUTE_BY:
            features.add(col)
            #print('Missing values imputed: {}'.format(col))
            impute_missing(train, col, IMPUTE_BY[col])
            impute_missing(test, col, IMPUTE_BY[col])
        if col in OTHERS and NORMALIZE == True:
            #print('Normalizing: {}'.format(col))
            features.add(col)
            train[col] = normalize(train.loc[:, [col]], axis=0)
            test[col] = normalize(test.loc[:, [col]],axis=0)
    return train, test, features


def model_ready(clean_train, clean_test, features):
    '''
    '''
    features = list(features)
    x_train = clean_train.filter(features)
    y_train = clean_train.filter(PREDICTED)
    x_test = clean_test.filter(features)
    y_test = clean_test.filter(PREDICTED)
    return x_train, y_train, x_test, y_test


def split_data(df, predicted='label', test_size=0.3, seed=1):
    '''
    Splits data into train and test
    '''
    X = df.drop('label', axis=1)
    Y = df[predicted]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return x_train, x_test, y_train, y_test


def join_df(df1, df2):
    return pd.concat([df1, df2], axis=1, join='inner')
