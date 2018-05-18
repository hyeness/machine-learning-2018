from magic_loop_fcns import *

##################
# PRE PROCESSING #
##################

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

# thyme
THYME = ['date_posted']

# drop later
ID = ['teacher_acctid', 'schoolid', 'school_ncesid']

OTHERS = ['fulfillment_labor_materials', 'total_price_excluding_optional_support',
          'total_price_including_optional_support', 'students_reached']

DATE = ['date_posted']

PREDICTED = ['fully_funded']

IMPUTE_BY = {'students_reached': 'mean'}

NORMALIZE = True


##################
#      MODEL     #
##################


# which parameter grid do we want to use (test, small, large)
GRID = TEST_GRID

# which variable to use for prediction_time
DATE_COL = 'date_posted'

# list of classifier models to run
TO_RUN = ['GB','RF','DT','KNN','LR','NB']

# cutoff and validation date pairs
CUTOFF_VAL_PAIRS = [('2011-06-30', '2011-12-31'), ('2011-12-31', '2012-06-30'),
                    ('2012-06-30', '2012-12-31'), ('2012-12-31', '2013-06-30'),
                    ('2013-06-30', '2013-12-31')]


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
