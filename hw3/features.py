from magic_loop_fcns import *

##################
# PRE PROCESSING #
##################

# convert bad booleans from t,f to 1,0
BINARY = ['fully_funded', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns',
         'school_kipp', 'school_charter_ready_promise', 'teacher_teach_for_america',
         'teacher_ny_teaching_fellow',
         'eligible_double_your_impact_match', 'eligible_almost_home_match']

# dummify categorical variables using top 5 most frequent values, other or missing otherwise
CATEGORICAL = ['school_metro', 'primary_focus_subject', 'primary_focus_area',
           'secondary_focus_subject', 'secondary_focus_area',
           'resource_type', 'grade_level', 'school_state', 'school_zip',
           'teacher_prefix', 'fulfillment_labor_materials']

# ignore for now
GEOGRAPHICAL = ['school_latitude', 'school_longitude', 'school_city', 'school_state',
                'school_county', 'school_state', 'school_zip', 'school_district']

# thyme
THYME = ['date_posted']

# drop later
ID = ['teacher_acctid', 'schoolid', 'school_ncesid']

OTHERS = ['total_price_excluding_optional_support',
          'total_price_including_optional_support', 'students_reached']

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
TO_RUN = ['AB', 'RF', 'DT', 'LR', 'NB', 'KNN', 'BAG']

# cutoff and validation date pairs
CUTOFF_VAL_PAIRS = [('2011-06-30', '2011-12-31'), ('2011-12-31', '2012-06-30'),
                    ('2012-06-30', '2012-12-31'), ('2012-12-31', '2013-06-30'),
                    ('2013-06-30', '2013-12-31')]

FEATURES = {'date_posted_month',
             'eligible_almost_home_match',
             'eligible_double_your_impact_match',
             'fulfillment_labor_materials_is_35.0',
             'fulfillment_labor_materials_is_nan',
             'fulfillment_labor_materials_is_others',
             'grade_level_is_Grades 3-5',
             'grade_level_is_Grades 6-8',
             'grade_level_is_Grades 9-12',
             'grade_level_is_Grades PreK-2',
             'grade_level_is_nan',
             'grade_level_is_others',
             'primary_focus_area_is_Applied Learning',
             'primary_focus_area_is_Literacy & Language',
             'primary_focus_area_is_Math & Science',
             'primary_focus_area_is_Music & The Arts',
             'primary_focus_area_is_Special Needs',
             'primary_focus_area_is_nan',
             'primary_focus_area_is_others',
             'primary_focus_subject_is_Applied Sciences',
             'primary_focus_subject_is_Literacy',
             'primary_focus_subject_is_Literature & Writing',
             'primary_focus_subject_is_Mathematics',
             'primary_focus_subject_is_Special Needs',
             'primary_focus_subject_is_nan',
             'primary_focus_subject_is_others',
             'resource_type_is_Books',
             'resource_type_is_Other',
             'resource_type_is_Supplies',
             'resource_type_is_Technology',
             'resource_type_is_Trips',
             'resource_type_is_nan',
             'resource_type_is_others',
             'school_charter',
             'school_charter_ready_promise',
             'school_kipp',
             'school_magnet',
             'school_metro_is_nan',
             'school_metro_is_others',
             'school_metro_is_rural',
             'school_metro_is_suburban',
             'school_metro_is_urban',
             'school_nlns',
             'school_state_is_CA',
             'school_state_is_FL',
             'school_state_is_NC',
             'school_state_is_NY',
             'school_state_is_TN',
             'school_state_is_nan',
             'school_state_is_others',
             'school_year_round',
             'school_zip_is_33610.0',
             'school_zip_is_33626.0',
             'school_zip_is_33647.0',
             'school_zip_is_38118.0',
             'school_zip_is_38128.0',
             'school_zip_is_nan',
             'school_zip_is_others',
             'secondary_focus_area_is_Applied Learning',
             'secondary_focus_area_is_History & Civics',
             'secondary_focus_area_is_Literacy & Language',
             'secondary_focus_area_is_Math & Science',
             'secondary_focus_area_is_Music & The Arts',
             'secondary_focus_area_is_nan',
             'secondary_focus_area_is_others',
             'secondary_focus_subject_is_ESL',
             'secondary_focus_subject_is_Literacy',
             'secondary_focus_subject_is_Literature & Writing',
             'secondary_focus_subject_is_Mathematics',
             'secondary_focus_subject_is_Special Needs',
             'secondary_focus_subject_is_nan',
             'secondary_focus_subject_is_others',
             'students_reached',
             'teacher_ny_teaching_fellow',
             'teacher_prefix_is_Mr.',
             'teacher_prefix_is_Mrs.',
             'teacher_prefix_is_Ms.',
             'teacher_prefix_is_nan',
             'teacher_prefix_is_others',
             'teacher_teach_for_america',
             'total_price_excluding_optional_support',
             'total_price_including_optional_support'}



def feature_importance(X, y, top_k):
    '''
    identify important features using a random forest
    This is based on sklearn example code:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
    '''

    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Feature Ranking
    importance = pd.DataFrame(columns=['feature', 'importance'])

    for f in range(0, top_k):
        importance.loc[f+1] = [X.columns[indices[f]], importances[indices[f]]]

    return importance


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
