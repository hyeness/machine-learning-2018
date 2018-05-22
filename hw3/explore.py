import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
from scipy import stats


def corr_matrix(df, col, save=False, filename=None):
    '''
    Plot correlation between all variables
    '''
    corr = df[col].corr()
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.title('Correlation Matrix')
    if save:
        plt.savefig(filename)
    plt.show()


def density_plot(df, column, log_=False, ignore_null=True):
    '''
    Plot density of variable
    Refuses to plot Missing Values
    '''

    if ignore_null:
        x = df[column].dropna()
    else:
        x = df[column]

    if log_:
        sns.distplot(x.apply(logify))
        plt.title('Log {}'.format(column))
    else:
        sns.distplot(x)
        plt.title(column)

    plt.show()

def logify(x):
    if x > 0:
        return log(x)
    else:
        return 0



def plot_hist(df, col, label, top_k, sort=True):
    '''
    plots histogram of column
    '''

    if sort:
        if top_k:
            hist_idx = df[col].value_counts().head(top_k)
        else:
            hist_idx = df[col].value_counts()
    else:
        hist_idx = df[col].value_counts(sort=False)

    graph = sns.countplot(x=col, saturation=1, data=df, order=hist_idx.index)
    plt.ylabel('Number in Sample')
    plt.xlabel(label)
    plt.title('Distribution of {}'.format(label))
    plt.show()



def count_nulls(df):
	'''
	Return number of null values for each column
	'''
	return df.isnull().sum()


def plot_correlations(df, title):
	'''
	Plot heatmap of columns in dataframe
	'''
	ax = plt.axes()
	corr = df.corr()
	sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax)
	ax.set_title(title)


def plot_dist(df, col, title, normal=True):
	'''
	Plot distribution of a column
	'''
	ax = plt.axes()
	if normal:
		sns.distplot(df[col], fit=stats.norm, kde=False, ax=ax)
	else:
		sns.distplot(df[col], kde=False, ax=ax)
	ax.set_title(title)
