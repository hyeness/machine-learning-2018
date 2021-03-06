import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from os import path
from math import log

######################
#      READ FILE     #
######################


DATA_FILENAME = 'data/credit-data.csv'
DATA_DICTIONARY = 'data/Data Dictionary.xls'

def read_file(filename, index=None):
    '''
    Reads file into pandas df
    '''
    ext = path.split(filename)[-1].split('.')[-1]

    if ext == 'csv':
        return pd.read_csv(filename, index_col=index)
    elif ext == 'xls':
        return pd.read_excel(filename, index_col=index)
    else:
        print("Not a valid filetype")


def create_data_dic(ddf):
    '''
    Converts data dictionary df into python dictionary
    '''
    dic = {}
    for i, row in ddf.iterrows():
        dic[row['Variable Name']] = row['Description'].title()
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


######################
#    EXPLORE DATA    #
######################


def corr_matrix(df):
    '''
    Plot correlation between all variables
    '''
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()


def density_plot(df, column, dic, log_=False):
    '''
    Plot density of variable
    Refuses to plot Missing Values
    '''
    if log_:
        sns.distplot(df[column].apply(logify))
        plt.title('Log {}'.format(dic[column]))
    else:
        sns.distplot(df[column])
        plt.title(dic[column])
    plt.show()

def logify(x):
    if x > 0:
        return log(x)
    else:
        return 0


def plot_hist(df, col, label, sort=True):
    '''
    plots histogram of column
    '''
    if sort:
        hist_idx = df[col].value_counts()
    else:
        hist_idx = df[col].value_counts(sort=False)

    graph = sns.countplot(x=col, saturation=1, data=df, order=hist_idx.index)
    plt.ylabel('Number in Sample')
    plt.xlabel(label)
    plt.title('Distribution of {}'.format(label))
    plt.show()


#######################
#  GENERATE FEATURES  #
#######################


def discretize_var(df, col, inc):
    '''
    Converts numeric variables into categorical
    i.e. age into age bins
    '''
    df[col] = df[col].astype(int)
    lb = df[col].min() // inc
    ub = df[col].max() // inc + 2
    boundaries = range(lb * inc, ub * inc, inc)
    col_bin = "{}_bin".format(col)
    df[col_bin] = pd.cut(df[col], bins=boundaries,
                         labels=range(len(boundaries)-1),
                         include_lowest=True, right=True)

def


######################
#  BUILD CLASSIFIER  #
######################

FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
       'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']


def split_data(df, predicted='SeriousDlqin2yrs', features=FEATURES,
                test_size=0.3, seed=1):
    '''
    Splits data into train and test
    '''
    X = df.filter(features, axis=1)
    Y = df[predicted]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return x_train, x_test, y_train, y_test


def select_features(data_set, features=FEATURES):
    '''
    filter data on features provided
    '''
    return data_set.filter(features)


def knn_models(x_train, y_train, x_test, y_test,
               ks=[1, 5, 10, 20, 50, 100], metric='minkowski', threshold=0.5):
    '''
    Returns KNN model with the highest accuracy score
    '''
    top_model = None
    score = 0

    model_params = []

    for k in ks:
        for wfn in ['uniform', 'distance']:
            for p in range(1, 6):
                knn = KNeighborsClassifier(n_neighbors=k,
                                           metric=metric,
                                           p=p,
                                           weights=wfn)

                knn.fit(x_train, y_train)
                pred = get_prediction(knn, x_test, y_test, threshold)
                acc = accuracy_score(y_test, pred[1])
                print(k, metric, knn.p, wfn, acc)
                model_params.append((k, metric, knn.p, wfn, acc))

                if acc > score:
                    score = acc
                    top_model = knn

    eval_df = pd.DataFrame(model_params, columns=('k', 'metric', 'p', 'weight_fcn', 'accuracy_score'))

    return top_model, eval_df


def get_prediction(model, x_test, y_test, threshold=0.5):
    '''
    takes a knn model fit on the training data
    gets prediction on testing data
    '''
    #model.fit(x_train, y_train)
    prob_true = model.predict_proba(x_test)[:, 1]
    pred = [classify(p, threshold) for p in prob_true]
    return prob_true, pred


def test_results(model, x_test, y_test, threshold=0.5):
    '''
    puts results in dataframe with all features used in testing
    results include probability of true, classified prediction, and true value
    '''
    result = x_test.copy()
    prob_true, pred = get_prediction(model, x_test, y_test, threshold)
    result['actual'] = y_test
    result['predicted'] = pred
    result['prob_true'] = prob_true
    return result


def classify(x, threshold):
    '''
    helper function classifying prediction as 1 if prob > threshold
    0 otherwise
    '''
    if x > threshold:
        return 1
    else:
        return 0


def just_predict(model, x_train, y_train, x_test, y_test):
    '''
    fits model on training data,
    makes prediction on x_test using knn.predict,
    returns results df cont. both true and predicted y values
    '''
    model.fit(x_train, y_train)
    result = x_test.copy()
    result['actual'] = y_test
    result['predicted'] = model.predict(x_test)
    return result



#########################
#  EVALUATE CLASSIFIER  #
#########################


def validate(y_true, y_predicted):
    '''
    prints accuracy of model and returns confusion matrix
    '''
    print('Accuracy Score: {}'.format(accuracy_score(y_true, y_predicted)))
    cm = confusion_matrix(y_true, y_predicted)
    actual = ['actual_no', 'actual_yes']
    predicted = ['predicted_no', 'predicted_yes']
    confusion = pd.DataFrame(cm, index=actual, columns=predicted)
    return confusion


def F1_score(confusion_matrix):
    '''
    calculates precision, recall, and F1 score (harmonic mean)
    '''
    true_positive = confusion_matrix['predicted_yes']['actual_yes']
    true_negative = confusion_matrix['predicted_no']['actual_no']
    false_positive = confusion_matrix['predicted_yes']['actual_no']
    false_negative = confusion_matrix['predicted_no']['actual_yes']

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print('Precision: {} \nRecall: {}'.format(precision, recall))

    F1 = 2 * (precision * recall) / (precision + recall)
    return F1


#if __name__ == "__main__":
    #main()
