import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from urllib.request import urlopen


#####################
#    QUESTION 1     #
#####################

GRAFFITI = 'Data/graffiti.csv'
ALLEY_LIGHTS = 'Data/alley_lights.csv'
VACANT = 'Data/vacant.csv'

def load_data(FILENAME, start='01/01/2017', end='12/31/2017'):

    df = pd.read_csv(FILENAME)
    start_date = datetime.strptime(start, "%m/%d/%Y")
    end_date = datetime.strptime(end, "%m/%d/%Y")

    if FILENAME == VACANT:
        date_col = 'DATE SERVICE REQUEST WAS RECEIVED'
    else:
        date_col = 'Creation Date'

    df['date'] = pd.to_datetime(df[date_col], format="%m/%d/%Y")
    df = df[(df.date >= start) & (df.date <= end)]
    df = df.drop(['date'], axis=1)

    return df

def combine_data(start, end):

    graffiti = load_data(GRAFFITI, start, end)
    alley_lights = load_data(ALLEY_LIGHTS, start, end)
    vacant = load_data(VACANT, start, end)

    for df in [graffiti, alley_lights, vacant]:
        df.columns = [x.title() for x in df.columns]

    vacant = clean_vacant(vacant)
    combined = pd.concat([graffiti, alley_lights, vacant])
    combined = compute_response_time(combined)

    for col in ['Ward', 'Zip Code', 'Police District', 'Community Area']:
        combined = combined.drop(combined.index[combined[col] == 0])

    combined = create_month_bins(combined)

    return combined

def clean_vacant(vacant):

    vacant = vacant.rename(columns={'Service Request Type': 'Type Of Service Request',
                                    'Date Service Request Was Received': 'Creation Date'})
    vacant['Address Street Number'] = vacant['Address Street Number'].real.astype(int).astype(str)
    vacant['Street Address'] = vacant['Address Street Number'].astype(str) + ' ' + vacant['Address Street Direction'].astype(str) + ' ' + vacant['Address Street Name'].astype(str) + ' ' + vacant['Address Street Suffix'].astype(str)
    drop_cols = ['Address Street Number', 'Address Street Direction', 'Address Street Name', 'Address Street Suffix']
    vacant = vacant.drop(drop_cols, axis=1)
    return vacant

def compute_response_time(combined):

    combined['Creation Date'] = pd.to_datetime(combined['Creation Date'], format="%m/%d/%Y")
    combined['Completion Date'] = pd.to_datetime(combined['Completion Date'], format="%m/%d/%Y")
    combined['Response Time'] = combined['Completion Date'] - combined['Creation Date']

    return combined


def col_to_hist(df, col, label, vertical=False, sort=True):

    plt.figure(figsize=(9,6))

    if sort:
        hist_idx = df[col].value_counts()
    else:
        hist_idx = df[col].value_counts(sort=False)

    if vertical:
            graph = sns.countplot(y=col, saturation=1, data=df, order=hist_idx.index)
            plt.ylabel(label)
            plt.xlabel('Number of 311 Requests')
    else:
        graph = sns.countplot(x=col, saturation=1, data=df, order=hist_idx.index)
        plt.xlabel(label)
        plt.ylabel('Number of 311 Requests')

    plt.title('Request Counts by {}'.format(col))
    plt.show()

def create_month_bins(df):
    df['Creation Month'] = df["Creation Date"].apply(lambda x: x.month)
    return df

def scatter_map(df, color_by='Type Of Service Request'):
    plt.figure()
    graph = sns.lmplot(x="Longitude", y="Latitude", data=df, fit_reg=False, hue=color_by, scatter_kws={"alpha":0.3,"s":10})
    plt.title("Spatial Distribution by {}".format(color_by))
    plt.show()

def scatter(df, request_type, dem_var):
    plt.figure()
    graph = sns.lmplot(x="count", y="mean", data=df, fit_reg=False, scatter_kws={"alpha":0.3,"s":10})
    plt.xlabel('Average {}'.format(dem_var))
    plt.ylabel('Number of Requests')
    plt.title("Number of {} Requests by {}".format(request_type, dem_var))
    plt.show()

#####################
#    QUESTION 2     #
#####################

CENSUS_API_KEY = 'f432c22b5284cc5583f011c7b027a78db588402c'

def abridged_data():
    last_three= combine_data(start='10/01/2017', end='12/31/2017')
    last_three = last_three[last_three['Type Of Service Request'] != 'Graffiti Removal']
    last_three = last_three.dropna(subset=['Latitude', 'Longitude'])
    return last_three

def get_block(row):

    lat = row['Latitude']
    long = row['Longitude']

    FIPS_url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&showall=true&format=json'.format(str(lat),str(long))

    try:
        response = urlopen(FIPS_url)
        FIPS = response.read()
        FIPS = json.loads(FIPS)
        FIPS_block = FIPS['Block']['FIPS']
        #state = FIPS_block[0:2]
        #county = FIPS_block[2:5]
        tract = FIPS_block[5:11]

        return tract

    except:
        print(FIPS_url)


def generate_FIPS(df):

    tract = []
    for i, row in df.iterrows():
        tract.append(get_block(row))

    df['FIPS Block Number'] = pd.Series(tract, index=df.index)

    return df


PCT_YT = 'DP05_0032PE'
ENGLISH_ONLY = 'DP02_0111PE'
INCOME = 'DP03_0051E'
FAMILY_SIZE = 'DP02_0016E'


def scrape_census_tract():
    variables = [PCT_YT, ENGLISH_ONLY, INCOME, FAMILY_SIZE]
    variables = ",".join(variables)
    url = 'https://api.census.gov/data/2015/acs5/profile?get=NAME,{}&for=tract:*&in=state:17+county:031&key={}'.format(variables, CENSUS_API_KEY)
    r = requests.get(url)
    if r.status_code == 200:
        acs = r.json()
    cols = ['Tract Name', 'Percent White', 'Percent English Only', 'Income', 'Family Size', 'State', 'County', 'Tract']
    census = pd.DataFrame(acs[1:], columns=cols)

    return census.set_index('Tract')


def join_df(df1, df2):
    '''
    join 2 dfs using index
    '''
    result = pd.concat([df1, df2], axis=1, join='inner')
    result = result.replace('-', np.nan)
    dem_vars = ['Percent White', 'Percent English Only', 'Income', 'Family Size']
    for var in dem_vars:
        result[var] = result[var].astype(float)

    return result

def demographic_stats(df, request_type, var):
    '''
    '''
    filtered = df[(df['Type Of Service Request'] == request_type)]
    summary = filtered[var].groupby(filtered['Zip Code']).describe()
    summary = summary.filter(['count', 'mean'])

    return summary.sort_values(by='count', ascending=False).head(10)
