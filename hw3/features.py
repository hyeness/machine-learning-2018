from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pdb


c_values = [10**i for i in range(-2, 3)] # pick the C values

def cv_logistic_regression(X_train, y_train, cvals=c_values, n_splits=5):
    kf = KFold(n_splits=n_splits)
    results = {} # dict of model params -> model performance over the KFold cross validation
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_train)):
        x_split_train, x_split_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_split_train, y_split_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

        for c in cvals:
            for p in ['l1', 'l2']:
                logreg = LogisticRegression(C=c, penalty=p)
                #pdb.set_trace()
                logreg.fit(x_split_train, y_split_train)
                y_pred = logreg.predict(x_split_test)
                model_key = (c, p) # this will be a longer tuple for things with more parameters
                results[(c, p)] =  results.get((c, p), 0) + f1_score(y_split_test, y_pred) / splits

    for model, model_perf in results.items():
        print("Model with params: {} | F1: {:.2f}".format(model, model_perf))
    return results
