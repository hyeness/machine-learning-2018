from magic_loop_fcns import *
from preprocess import *
from features import *
from datetime import timedelta

def temporal_split(df, col, cutoff_date, validation_date):
    '''
    returns train, test pairs
    '''
    train_end = datetime.strptime(cutoff_date, '%Y-%m-%d')
    test_end = datetime.strptime(validation_date, '%Y-%m-%d')
    train = df[df[col] <= train_end]
    test = df[(df[col] > train_end) & (df[col] <= test_end)]

    return train, test


def model_ready(clean_train, clean_test, features):
    '''
    '''
    features = list(features)
    x_train = clean_train.filter(features)
    y_train = clean_train.filter(PREDICTED)
    x_test = clean_test.filter(features)
    y_test = clean_test.filter(PREDICTED)
    return x_train, y_train, x_test, y_test


# adapted from Rayid's magic loop
def tempura_validation_loop(df, cv_pairs, grid_size, to_run, filename, use_importance=False):
    '''
    '''

    # define dataframe to write results to
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'validation_date',
                                        'train_set_size', 'validation_set_size', 'baseline',
                                        'precision_at_5','precision_at_10','precision_at_20', 'precision_at_30', 'precision_at_50',
                                        'recall_at_5','recall_at_10','recall_at_20', 'recall_at_30', 'recall_at_50',
                                        'auc-roc','time_elapsed', 'features_used'))

    for c, v in cv_pairs:
        print('CUTOFF: {} VALIDATION: {}'.format(c, v))

        train, test = temporal_split(df, DATE_COL, c, v)

        X_train, y_train, X_test, y_test = model_ready(*pre_process(train, test))
        features_used = 'ALL'
        if use_importance:
            important = list(feature_importance(X_train, y_train, 20)['feature'])
            features_used = important

            print(important)
            X_train = X_train.filter(important)
            X_test = X_test.filter(important)

        for i, clf in enumerate([CLASSIFIERS[x] for x in to_run]):
            print(to_run[i])
            params = grid_size[to_run[i]]
            print(params)
            for p in ParameterGrid(params):
                try:
                    start_time = time.time()
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train.values.ravel()).predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))
                    end_time = time.time()
                    tot_time = end_time - start_time
                    print(tot_time)

                    precision_5, accuracy_5, recall_5 = scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
                    precision_10, accuracy_10, recall_10 = scores_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
                    precision_20, accuracy_20, recall_20 = scores_at_k(y_test_sorted,y_pred_probs_sorted,20.0)
                    precision_30, accuracy_30, recall_30 = scores_at_k(y_test_sorted,y_pred_probs_sorted,30.0)
                    precision_50, accuracy_50, recall_50 = scores_at_k(y_test_sorted,y_pred_probs_sorted,50.0)

                    results_df.loc[len(results_df)] = [to_run[i], clf, p, v,
                                                    y_train.shape[0], y_test.shape[0],
                                                    scores_at_k(y_test_sorted, y_pred_probs_sorted,100.0)[1],
                                                    precision_5, precision_10, precision_20, precision_30, precision_50,
                                                    recall_5, recall_10, recall_20, recall_30, recall_50,
                                                    roc_auc_score(y_test, y_pred_probs),
                                                    tot_time, features_used]

                    plot_precision_recall_n(y_test, y_pred_probs, clf, False)

                except IndexError:
                    print('Error')
                    continue
            results_df.to_pickle(filename)
    return results_df


def split_data(df, predicted='label', test_size=0.3, seed=1):
    '''
    Splits data into train and test
    '''
    X = df.drop('label', axis=1)
    Y = df[predicted]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return x_train, x_test, y_train, y_test


def kfold_validation(df, n_splits=5):
    '''
    '''

    X_train, X_test, y_train, y_test = model_ready(train, test)

    kf = KFold(n_splits)
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_train)):
        columns = ['fold_num', 'model_type','clf', 'parameters', 'baseline',
                 'precision_5', 'accuracy_5', 'recall_5',
                 'precision_10', 'accuracy_10', 'recall_10',
                 'precision_20', 'accuracy_20', 'recall_20',
                 'auc-roc', 'y_pred_probs', 'runtime']
        results = pd.DataFrame(columns=columns)
        x_split_train, x_split_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_split_train, y_split_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
        classifiers_loop(x_split_train, x_split_test, y_split_train, y_split_test, results_df, k)

        return results



def classifiers_loop(X_train, X_test, y_train, y_test, results_df, k):
    '''
    '''

    for i, clf in enumerate([CLASSIFIERS[x] for x in TO_RUN]):
        #print(TO_RUN[i])
        params = GRID[TO_RUN[i]]
        for p in ParameterGrid(params):
            try:
                start_time = time.time()
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train.values.ravel()).predict_proba(X_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))
                end_time = time.time()
                tot_time = end_time - start_time

                baseline = (y_train.sum() + y_test.sum())/ (y_train.shape[0] + y_test.shape[0])
                precision_5, accuracy_5, recall_5 = scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
                precision_10, accuracy_10, recall_10 = scores_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
                precision_20, accuracy_20, recall_20 = scores_at_k(y_test_sorted,y_pred_probs_sorted,20.0)

                if k != -1:
                    results_df.loc[len(results_df)] = [k, to_run[i], clf, p, baseline,
                                               precision_5, accuracy_5, recall_5,
                                               precision_10, accuracy_10, recall_10,
                                               precision_20, accuracy_20, recall_20,
                                               roc_auc_score(y_test, y_pred_probs),
                                               y_pred_probs, tot_time]

                    #plot_precision_recall_n(y_test,y_pred_probs,clf)
            except IndexError:
                print('Error')
                continue

    #return results_df
