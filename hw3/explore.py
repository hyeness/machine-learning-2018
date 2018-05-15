import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log



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
