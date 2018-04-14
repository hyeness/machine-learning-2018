import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


######################
#      READ FILE     #
######################


DATA_FILENAME = 'data/credit-data.csv'
DATA_DICTIONARY = 'data/Data Dictionary.xls'

def read_file(filename, filetype, index=None):
    if filetype == 'csv':
        return pd.read_csv(filename, index_col=index)
    if filetype == 'excel':
        return pd.read_excel(filename, index_col=index)


def create_data_dic(ddf):
    dic = {}
    for i, row in ddf.iterrows():
        dic[row['Variable Name']] = row['Description']
    return dic

######################
#  PRE-PROCESS DATA  #
######################


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


def cap_value(x, cap):
    if x > cap:
        return cap
    else:
        return x

def cap_outlier(df, column):
    cap = df[column].quantile(.999)
    df[column] = df[column].apply(cap_value, args=(cap,))


######################
#  DATA EXPLORATION  #
######################


def corr_matrix(df):
    '''
    Plot correlation between all variables
    '''
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()


def density_plot(df, column, dic):
    '''
    Plot density of variable
    '''
    sns.kdeplot(df[column], shade=True)
    plt.title(dic[column])
    plt.show()



#######################
#  GENERATE FEATURES  #
#######################

def discretize_var(df, col, inc):
    '''
    '''
    df[col] = df[col].astype(int)
    lb = df[col].min() // inc
    ub = df[col].max() // inc + 2
    boundaries = range(lb * inc, ub * inc, inc)
    col_bin = "{}_bin".format(col)
    df[col_bin] = pd.cut(df[col], bins=boundaries,
                         labels=range(len(boundaries)-1),
                         include_lowest=True, right=True)


######################
#  BUILD CLASSIFIER  #
######################


def split_data(df, predicted='SeriousDlqin2yrs', features, test_size=0.3):
    X = df.filter(features, axis=1)
    Y = df[predicted]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test

knn = KNeighborsClassifier(n_neighbors=10,
                           metric='minkowski',
                           metric_params={'p': 3})

def run_model(model, threshold=0.5):
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 0]

    result = x_test.copy()
    result['predicted_prob'] = list(prob)
    result['classify'] = result['predicted_prob'].apply(classify, args=(threshold,))

    return result

def classify(x, threshold):
    if x > threshold:
        return True
    else:
        return False



df = read_file(DATA_FILENAME, 'csv', 'PersonID')
df.head()
check_missing(df)
impute_missing(df, 'NumberOfDependents', 'median')
impute_missing(df, 'MonthlyIncome', 'mean')
#df = discretize_var(df, 'age', 10)
df.MonthlyIncome.isnull().any()
df.NumberOfDependents = df.NumberOfDependents.astype(int)
df.dtypes
cap_outlier(df, 'MonthlyIncome')
df.MonthlyIncome.max()
sns.distplot(df.MonthlyIncome)
% matplotlib inline
corr_matrix(df)
