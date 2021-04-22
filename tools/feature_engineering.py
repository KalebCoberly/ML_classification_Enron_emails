#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

###
### FIXME: Regression can't handle NaN input, though it could elsewhere.
###
# ### Also flag if ratio above it's regression.

# def flag_reg_error(y_sr, X_sr, index=None, pair_name=None,
#                    sk_reg=LinearRegression, **kwargs):
#     '''Flag whether a point falls above or below the regression.
#     Parameters:
#         y_sr: (pandas.Series, float) of y values.
#         x_sr: (pandas.Series, float) of x values.
#         index: (pandas.Series.index) to assign flagged_sr. If None (default),
#             use y_sr.index.
#         pair_name: (string) to name the return series ("__reg_error" appended);
#             If None (default), y_sr.name + "__" + x_sr.name.
#         sk_reg: (sklearn regression function) to fit and predict.
#         **kwargs: (list) keyword arguments to pass to sk_reg.
#     Return:
#         flagged_sr: (pandas.Series, bool) flags whether error is above 0.'''
    
#     flagged_sr = None #pd.Series()
    
#     if pair_name is None:
#         pair_name = y_sr.name + '__' + X_sr.name
#     if index is None:
#         index = y_sr.index
        
#     reg = sk_reg(**kwargs).fit(X=[X_sr], y=[y_sr])
#     pred = reg.predict(X=X_sr)
    
#     error_sr = y_sr - pred
#     flagged_sr = pd.Series(error_sr.apply(func=(lambda x: np.nan
#                                                 if x == np.nan or x == np.inf \
#                                                 else x > 0)))
#     flagged_sr.index = index
#     flagged_sr.name = pair_name + '__reg_error'
    
#     return flagged_sr


def set_all_ratios(df, denoms, numers, flag_error=False):
    '''Create new features from ratios of pairs of features.
    Optionally create feature that flags if cartesion point falls above or
    below regression.
    Parameters:
        df: (pandas.DataFrame) selected columns must be numeric, float best.
        denoms: (list, str) names of columns to use as y in y/x.
        numers: (list, str) names of columns to use as x in y/x.
        flag_error: (bool) If True (default: False) create additional feature
            that flags whether point falls above regression.
    Return:
        ratio_subspace_df: (pandas.DataFrame) new features to add to dataset.
    '''
    ratio_subspace_df = None #pd.DataFrame()
    
    ### We want both ratios for each to get the 0s reciprocal to inf.
    for y in numers:
        for x in denoms:
            if y != x:
                ratio_sr = df[y] / df[x]
                ratio_sr.name = y + '_DivBy_' + x
                ratio_subspace_df = pd.concat(objs=[ratio_subspace_df,
                                                    ratio_sr],
                                              axis=1)
                
###
### FIXME: finish flag_error (regression not accepting NaNs, does elsewhere)
###
#                 if flag_error:
#                     ratio_subspace_df = pd.concat(objs=[ratio_subspace_df,
#                                             flag_reg_error(y_sr=df[y],
#                                                            X_sr=df[x])],
#                                            axis=1)
    
    return ratio_subspace_df


def quant_flag_all(df, quant_df, quants=[4, 5, 10]):
    '''For each feature, create a categorical feature for each quantile. Skip .
    Parameters:
        df: (pandas.DataFrame, float) dataset with all features to flag.
        quant_df: (pandas.DataFrame, float) dataset to set quantiles;
            X_train if flagging X_test.
        quants: (list, int) quantiles to use.
            [4, 5, 10] will create 3 new features per feature,
            i.e. a quartile factor, a quintile factor, and a decile factor.
    Return:
        flagged_subspace_df: (pandas.DataFrame, float) dataset of new features.
    '''
    flagged_subspace_df = None #pd.DataFrame()
    
    for feat in df.columns:        
        ### Create quant ranges, empty df with column names.
        q_ranges_dict = {}
        q_subspace_df = pd.DataFrame(dtype=float, index=df.index)
        for quant in quants:
            q = np.arange(start=0, stop=1, step=(1/quant))
            qn = np.quantile(a=quant_df[feat].dropna(), q=q)
            q_ranges = np.append(arr=(np.append(arr=[-np.inf],
                                                values=qn)),
                                 values=[np.inf])
            q_name = feat + '_q_' + str(quant)
            q_ranges_dict[q_name] = q_ranges
            q_sr = pd.Series(dtype=float, name=q_name)
            q_subspace_df = pd.concat(objs=[q_subspace_df, q_sr], axis=1)
            
            # Set flags.
            for i, x in enumerate(df[feat]):
                if not np.isnan(x):
                    ### Populate each quant column.
                    for q_name, q_ranges in q_ranges_dict.items():
                        for j, low_q in enumerate(q_ranges):
                            if (x > low_q) and (x <= q_ranges[j + 1]):
                                q_subspace_df[q_name][df.index[i]] = j + 1
                                break
                else:
                    q_subspace_df[q_name][df.index[i]] = np.nan
        
        flagged_subspace_df = pd.concat(objs=[flagged_subspace_df, q_subspace_df], axis=1)

    return flagged_subspace_df.astype(float)


def out_flag_all(df, quant_df):
    '''For each feature, create a categorical feature to flag outliers, -1 for low outliers, 1 for high, and 0
    for not an outlier.
    Parameters:
        df: (pandas.DataFrame, float) dataset with all features to flag.
        quant_df: (pandas.DataFrame, float) dataset to set quantiles;
            X_train if flagging X_test.
    Return:
        flagged_subspace_df: (pandas.DataFrame, float) dataset of new features.
    '''
    flagged_subspace_df = None #pd.DataFrame()
    
    for feat in df.columns:
        ### Get outlier values, create empty series.
        quarts = np.quantile(a=quant_df[feat].dropna(), q=np.arange(0, 1, 1/3))
        IQRmult = 1.5 * (quarts[2] - quarts[0])
        outs = [(quarts[0] - IQRmult), (quarts[2] + IQRmult)]
        out_sr = pd.Series(dtype=int, name=(feat + '_outliers'))
        
        ### Set flags.
        for i, x in enumerate(df[feat]):
            if not np.isnan(x):
                if x < outs[0]:
                    out_sr[df[feat].index[i]] = -1
                elif x > outs[1]:
                    out_sr[df[feat].index[i]] = 1
                else:
                    out_sr[df[feat].index[i]] = 0
            else:
                out_sr[df[feat].index[i]] = np.nan
        
        flagged_subspace_df = pd.concat(objs=[flagged_subspace_df, out_sr], axis=1)

    return flagged_subspace_df.astype(float)


def flag_signs(df, suffix='_sign'):
    '''For each feature, create a feature that flags whether x > 0.
    Retain np.nans.
    Parameters:
        df: (pandas.DataFrame)
        suffix: (str) if not empty string, append suffix to
            column names.
    Returns
        flagged_df: (pandas.DataFrame, bool) of same shape as df.
    '''
    flagged_df = None #pd.DataFrame
    
    flagged_df = df.apply(func=(lambda col: col.apply(func=(lambda x: np.nan
                                                if np.isnan(x) else x > 0))))
    
    if suffix:
        new_cols_lst = []
        for col in flagged_df.columns:
            new_cols_lst.append(col + suffix)
        flagged_df.columns = new_cols_lst
        
    return flagged_df


def add_k_means_n(X_train, X_test, n_cluster_lst=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
    '''For each n_cluster, create n features that flag cluster membership.
    Parameters:
        X_train: (pandas.DataFrame) training set to fit and add features to.
        X_test: (pandas.DataFrame) testing set to add features to.
        n_cluster_lst: (list, int) n clusters to classify.
    Returns:
        train_cluster_subspace: (pandas.DataFrame, bool) new features to add to
            training set.
        test_cluster_subspace: (pandas.DataFrame, bool) new features to add to
            testing set.'''
    
    train_cluster_subspace = pd.DataFrame(index=X_train.index)
    test_cluster_subspace = pd.DataFrame(index=X_test.index)
    
    lamb = lambda x: x == i
    
    for n in n_cluster_lst:
        clf = KMeans(n_clusters=n, random_state=42).fit(X=X_train)
        train_sr = pd.Series(data=clf.predict(X=X_train),
                             index=X_train.index)
#         print('\ntrain_sr:\n', train_sr)
        test_sr = pd.Series(data=clf.predict(X=X_test),
                            index=X_test.index)
#         print('\ntest_sr\n:', test_sr)
        
        for i in np.arange(start=0, stop=n, step=1):
            col_name = 'k_means_' + str(n) + '_n_' + str(i)
            
            flag_sr = train_sr.apply(func=lamb)
#             flag_sr.index = X_train.index
            flag_sr.name = col_name
#             print('flag_sr.name:' + flag_sr.name)
            train_cluster_subspace = pd.concat(objs=[train_cluster_subspace,
                                                     flag_sr], axis=1)
            
            flag_sr = test_sr.apply(func=lamb)
#             flag_sr.index = X_test.index
            flag_sr.name = col_name
#             print('flag_sr.name:' + flag_sr.name)
            test_cluster_subspace = pd.concat(objs=[test_cluster_subspace,
                                                    flag_sr], axis=1)
    
    return train_cluster_subspace, test_cluster_subspace