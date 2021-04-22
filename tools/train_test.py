#!/usr/bin/python

from time import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def run_skl(method, X_train, y_train, X_test, y_test, print_perf=True,
            perf_series='', **kwargs):
    '''Train and test sklearn supervised ML models.
    ### Borrowed and modified from Udacity. ###
    ### Needs `from class_vis import prettyPicture` to graph. ###
    Parameters:
        method: callable sklearn supervised ML model.
        X_train: training feature matrix.
        y_train: training label 1D array. Series may throw warning.
        X_test: testing feature matrix.
        y_test: testing label 1D array. Series may throw warning.
        print_perf: {bool} True (default) prints performance metrics.
        per_series: {str} (default is empty) If not empty,
                    returns pandas Series with model and performance metrics.
                    String value is name of series.
        **kwargs: arguments to pass to method.
    Returns:
        clf: trained model.
        pred: array of predictions.
        conf: confusion matrix of test results.
        prf: sklearn.metrics.precision_recall_fscore_support results, weighted.
        perf_sr: pandas Series of model and metrics (everything but pred).
                 Must be binary class. Returns as None if perf_series False.
    '''
    clf = None
    pred = None
    conf = None
    prf = None
    perf_sr = None
    
    clf = method(**kwargs)

    t0 = time()
    clf = clf.fit(X=X_train, y=y_train)
    train_t = round(time()-t0, 3)
    
    t0 = time()
    pred = clf.predict(X=X_test)
    pred_t = round(time()-t0, 3)
    
    conf = confusion_matrix(y_true=y_test, y_pred=pred)
    prf = precision_recall_fscore_support(y_true=y_test, y_pred=pred)
    
###
### TO DO: fix from class_vis import prettyPicture
###
#     try:
#         prettyPicture(clf, X_test, y_test)
#     except NameError:
#         pass
    
    if print_perf:
        print(clf)
        print("Training time:", train_t, "s")
        print("Prediction time:", pred_t, "s")
        print("Confusion matrix:\n", conf)
        print("Precision, recall, f beta score, support:\n", prf, '\n')
    
    if perf_series != '':
        perf_sr = pd.Series(
            {
                'model': clf,
                'nonPOI_prec': prf[0][0],
                'POI_prec': prf[0][1],
                'nonPOI_rec': prf[1][0],
                'POI_rec': prf[1][1],
                'nonPOI_f': prf[2][0],
                'POI_f': prf[2][1],
                'nonPOI_sup': prf[3][0],
                'POI_sup': prf[3][1],
                't_neg': conf[0][0],
                'f_neg': conf[0][1],
                'f_pos': conf[1][0],
                't_pos': conf[1][1],
                'train_t': train_t,
                'predict_t': pred_t,
            },
            name=perf_series
        )
    
    return clf, pred, conf, prf, perf_sr


def get_base_perfs(base_perfs_dict, imp_sets_dict, clf_dict, y_train, y_test):
    
#     copy_base_perfs_dict = base_perfs_dict.deepcopy() #dict has no deepcopy
    
    for imp_name, imp_sets in imp_sets_dict.items():
        print('\n', imp_name)
        for clf_name, method in clf_dict.items():
            print('\n', clf_name)
            _, _, _, _, perf_sr = run_skl(method=method,
                                          X_train=imp_sets[0], y_train=y_train,
                                          X_test=imp_sets[1], y_test=y_test,
                                          perf_series=clf_name)
            base_perfs_dict[imp_name] \
                = base_perfs_dict[imp_name].append(perf_sr)
    
    return #copy_base_perfs_dict


def search_em_all(X_train, y_train, selectors, decomps, classifiers, pipe_verbose=True,
                  scoring='recall_weighted', n_jobs=-1):
    ### Try also 'jaccard_weighted'?
    gscv_dict = {}
    
    i = 0
    for selector in selectors:
        for decomp in decomps:
            for classifier in classifiers:
                pipe = None
                params = {}
                if decomps[decomp]:
                    pipe = Pipeline(steps=[
                        (selector, selectors[selector]['sel']),
                        (decomp, decomps[decomp]['dec']),
                        (classifier, classifiers[classifier]['clf'])
                    ], verbose=pipe_verbose)

                    params.update(selectors[selector]['params'])
                    params.update(decomps[decomp]['params'])
                    params.update(classifiers[classifier]['params'])
                else:
                    pipe = Pipeline(steps=[
                        (selector, selectors[selector]['sel']),
                        (classifier, classifiers[classifier]['clf'])
                    ], verbose=pipe_verbose)

                    params.update(selectors[selector]['params'])
                    params.update(classifiers[classifier]['params'])
                
                params_name = selector + '_' + decomp + '_' + classifier
                print('\n', i, params_name, '\n')
                
                gscv = GridSearchCV(estimator=pipe, param_grid=params,
                                    scoring=scoring, n_jobs=n_jobs, verbose=3)

                gscv_dict[params_name] = gscv.fit(X=X_train, y=y_train)
                print('\n', gscv_dict[params_name])
                print('\nbest_score_:', gscv_dict[params_name].best_score_)
                print('\nbest_params_:', gscv_dict[params_name].best_params_)
                
                i += 1
                
    return gscv_dict