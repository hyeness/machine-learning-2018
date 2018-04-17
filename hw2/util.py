import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


######################
#      READ FILE     #
######################


DATA_FILENAME = 'data/credit-data.csv'
DATA_DICTIONARY = 'data/Data Dictionary.xls'

def read_file(filename, filetype, index=None):
    '''
    Reads file into pandas df
    '''
    if filetype == 'csv':
        return pd.read_csv(filename, index_col=index)
    if filetype == 'excel':
        return pd.read_excel(filename, index_col=index)


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


def cap_value(x, cap):
    '''
    '''
    if x > cap:
        return cap
    else:
        return x

def cap_outlier(df, column):
    cap = df[column].quantile(.999)
    df[column] = df[column].apply(cap_value, args=(cap,))


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


def density_plot(df, column, dic):
    '''
    Plot density of variable
    '''
    sns.kdeplot(df[column], shade=True)
    plt.title(dic[column])
    plt.show()


def col_to_hist(df, col, label, sort=True):
    '''
    '''
    if sort:
        hist_idx = df[col].value_counts()
    else:
        hist_idx = df[col].value_counts(sort=False)

    graph = sns.countplot(y=col, saturation=1, data=df, order=hist_idx.index)
    plt.xlabel('Number in Sample')
    plt.ylabel(label)
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


######################
#  BUILD CLASSIFIER  #
######################

FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
       'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']


def split_data(df, predicted='SeriousDlqin2yrs', features=FEATURES, test_size=0.3):
    '''
    Splits data into train and test
    '''
    X = df.filter(features, axis=1)
    Y = df[predicted]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def knn_models(x_train, y_train, x_test, y_test, metric='minkowski'):
    '''
    Returns KNN model with the highest accuracy score
    '''
    top_model = None
    score = 0

    model_params = []

    for k in range(1, 11):
        for wfn in ['uniform', 'distance']:
            for p in range(1, 6):
                knn = KNeighborsClassifier(n_neighbors=k,
                                           metric=metric,
                                           p=p,
                                           weights=wfn)

                knn.fit(x_train, y_train)
                acc = accuracy_score(y_test, knn.predict(x_test))
                model_params.append((k, metric, knn.p, wfn, acc))

                if acc > score:
                    score = acc
                    top_model = knn

    eval_df = pd.DataFrame(model_params, columns=('k', 'metric', 'p', 'weight_fcn', 'accuracy_score'))

    return top_model, eval_df

#top_model = knn_models(x_train, y_train, x_test, y_test)
#result = just_predict(top_model, x_train, y_train, x_test, y_test)
#result.filter(['actual', 'predicted'])


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

def F1_score(val):
    '''
    calculates precision, recall, and F1 score (harmonic mean)
    '''
    true_positive = val['predicted_yes']['actual_yes']
    true_negative = val['predicted_no']['actual_no']
    false_positive = val['predicted_yes']['actual_no']
    false_negative = val['predicted_no']['actual_yes']

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print('Precision: {} \nRecall: {}'.format(precision, recall))

    F1 = 2 * (precision * recall) / (precision + recall)
    return F1


def run_model(model, x_train, y_train, x_test, y_test, threshold=0.5):
    model.fit(x_train, y_train)
    prob_true = model.predict_proba(x_test)[:, 0]
    result = x_test.copy()
    result['predicted_prob'] = prob_true
    result['actual'] = y_test
    result['predicted'] = result['predicted_prob'].apply(classify, args=(threshold,))

    return result

def classify(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0


#if __name__ == "__main__":
    #main()

def main():
    df = read_file(DATA_FILENAME, 'csv', 'PersonID')
    df.head()
    check_missing(df)
    impute_missing(df, 'NumberOfDependents', 'median')
    impute_missing(df, 'MonthlyIncome', 'median')
    #df = discretize_var(df, 'age', 10)
    df.MonthlyIncome.isnull().any()
    df.NumberOfDependents = df.NumberOfDependents.astype(int)
    df.dtypes
    cap_outlier(df, 'MonthlyIncome')
    df.MonthlyIncome.max()
    sns.distplot(df.MonthlyIncome)
    corr_matrix(df)
    x_train, x_test, y_train, y_test = split_data(df)
    x_train.head()
    top_model = knn_models_try(x_train, y_train, x_test, y_test)
    result = run_model(top_model, x_train, y_train, x_test, threshold=0.5)
    result.head()
    result.classify.value_counts()
    df.DebtRatio.describe()
    sns.distplot(df.DebtRatio)
    result.head()
    accuracy_score(y_test, result.classify)
    jp_result = just_predict(top_model, x_train, y_train, x_test, y_test)
    jp_result.actual.value_counts()
    accuracy_score(y_test, jp_result.prob_true)
