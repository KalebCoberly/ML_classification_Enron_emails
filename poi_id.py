#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer#, \
    # IterativeImputer # requires sklearn.experimental.enable_iterative_imputer

# from sklearn.feature_selection import SelectPercentile, SelectFromModel, f_classif, mutual_info_classif, chi2,\
#                                         SelectFpr, SelectFdr, RFECV
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, chi2
# from sklearn.decomposition import FastICA, IncrementalPCA, KernelPCA, PCA, TruncatedSVD

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

### My imports
sys.path.append('tools/')
from dos2unix import crlf_to_lf # Borrowed and modified from multiple sources.
from train_test import run_skl, get_base_perfs, search_em_all
from feature_engineering import set_all_ratios, quant_flag_all, out_flag_all, flag_signs, add_k_means_n

### Udacity imports (deprecated)
# from feature_format import featureFormat, targetFeatureSplit
# from tester import dump_classifier_and_data

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


##########################################################################################
### Udacity comments are in [].
### [Load the dictionary containing the dataset]
### Make the dict a dataframe because they're easier to work with.
data_df = None #pd.DataFrame()
fp = crlf_to_lf(f_in_path='data/final_project_dataset.pkl')
with open(fp, 'rb') as data_file:
    data_df = pd.DataFrame(pickle.load(data_file)).T

##########################################################################################
### [Task 1: Select what features you'll use.]
### Task 1: Clean up and select what features and subsets *not* to use.
### (Further feature selection will happen after feature engineering.)
print('Cleaning data.')
### Drop email_address because it's a signature.
data_df.drop(columns='email_address', inplace=True)
### Drop the TOTAL row.
data_df.drop(labels=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)

### Handle missing values here.
### Replacing 'NaN' with None had a weird result in which values from some
### rows were copied into the missing values of neighboring rows. No idea why.
### Using np.nan did not have that result as far as I can tell.
### But it is a float missing value and thus casts the column as float,
### or as object when other values are not floats.
data_df.replace(to_replace='NaN', value=np.nan, inplace=True)

### [features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".]
###    (if using featureFormat(), which I don't).

### All units are in USD.
fin_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
                'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments',
                'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
pay_features = fin_features[:10]
stock_features = fin_features[10:]
    
### Units are number of emails messages;
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

### Boolean, represented as integer.
POI_label = ['poi']

### The first feature must be "poi" if using featureFormat().
features_list = POI_label + fin_features + email_features

### Imputation recasts as float, but as object if left as bool, so set it to int for now.
data_df['poi'] = data_df['poi'].astype(dtype=int)

### Belfer's financial data is shifted one column to the right.
### Shift it one to the left, financial data only.
### Make total_stock_value np.nan for consistency until imputation, but could be 0.
### May remove this row for so many NaNs, but fix it now anyway.
data_df.loc[data_df.index == 'BELFER ROBERT', fin_features] \
    = data_df.loc[data_df.index == 'BELFER ROBERT', fin_features].shift(periods=-1, axis='columns',
                                                                        fill_value=np.nan)

### Bhatnagar's financial data is shifted one to the left.
### Shift it one to the right, financial data only.
### Make salary np.nan.
data_df.loc[data_df.index == 'BHATNAGAR SANJAY', fin_features] \
    = data_df.loc[data_df.index == 'BHATNAGAR SANJAY', fin_features].shift(periods=1, axis='columns',
                                                                           fill_value=np.nan)

### Set totals to sum of values where any values are not NaN.
### i.e. don't make 0 totals NaN, even though some NaN values may be included.
### Makes these rows consistent with other rows that include NaNs and numbers yet have a nonNaN total.
data_df.loc[~(data_df[pay_features].isna().all(axis='columns')), 'total_payments'] \
    = data_df[pay_features[:-1]].sum(axis='columns')
data_df.loc[~(data_df[stock_features].isna().all(axis='columns')), 'total_stock_value'] \
    = data_df[stock_features[:-1]].sum(axis='columns')

### Add one to Glisan's to_message to at least equal shared_receipt_with_poi.
data_df.loc['GLISAN JR BEN F', 'to_messages'] = 874

### Drop features that are too sparse.
drop_feats_lst = ['loan_advances']
data_df.drop(columns=drop_feats_lst, inplace=True)
fin_features = [feat for feat in fin_features if feat not in drop_feats_lst]
pay_features = [feat for feat in pay_features if feat not in drop_feats_lst]
stock_features = [feat for feat in stock_features if feat not in drop_feats_lst]
email_features = [feat for feat in email_features if feat not in drop_feats_lst]
features_list = [feat for feat in features_list if feat not in drop_feats_lst]

### Removed 'email' as signature upon loading.

### Drop persons who have NaN payment totals or NaN stock totals or NaN to_messages or NaN from_messages,
### and are missing 70% of their values.
### (Already made sure that all totals are not NaN if they have subvalues.)
nan_limit = 0.7 * len(data_df.columns)
sparse_records_idx_arr = \
    data_df.loc[data_df['total_payments'].isna() \
                | data_df['total_stock_value'].isna() \
                | data_df['to_messages'].isna() \
                | data_df['from_messages'].isna()]\
           .loc[data_df.isna().sum(axis='columns') > nan_limit]\
           .index.values
data_df.drop(labels=sparse_records_idx_arr, inplace=True)

### This leaves 123 records over 19 features.


### Make a quick baseline model for comparison.

### Alphabetize index before split for Udacity compatibility because that's what they'll do.
### I knew this, but missed it until the end. I'd decided from the start not to use their deprecated
### scripts based on dictionaries, specifically feature_format, and wrote my own using pandas. 
### The rest of my cleaning and engineering are based on a different split.
### My mistake. Rookie lesson learned: pay closer attention to what the legacy code does, especially 
### expected input/output structures.
data_df.sort_index(inplace=True)

### Split now for baseline model, but also before further processing, outlier removal, scaling, engineering,
### or else test set info leaks into training set.
### Even imputation could if using multivariate imputation or median.
### Decisions on how to treat the data should not be influenced by test set either.
X_train, X_test, y_train, y_test \
    = train_test_split(data_df[features_list[1:]], data_df[['poi']], test_size=.3, random_state=42)
### Some algorithms want 1D y data.
y_train_1d = np.ravel(y_train.astype(bool))
y_test_1d = np.ravel(y_test.astype(bool))

### Split train set again for a baseline model that won't touch the final test set.
X_train_base, X_test_base, y_train_base, y_test_base \
    = train_test_split(X_train, y_train, test_size=.3, random_state=42)
y_train_1d_base = np.ravel(y_train_base.astype(bool))
y_test_1d_base = np.ravel(y_test_base.astype(bool))

### Impute with 0.
imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0, copy=False)
imp_0 = imp_0.fit(X=X_train_base)
X_train_base_imp0 = pd.DataFrame(data=imp_0.transform(X=X_train_base), columns=X_train_base.columns,
                                 index=X_train_base.index)
X_test_base_imp0 = pd.DataFrame(data=imp_0.transform(X=X_test_base), columns=X_test_base.columns,
                                index=X_test_base.index)

### For metrics dataframe if you want to save and inspect them.
ordered_cols_lst = ['nonPOI_prec', 'POI_prec', 'nonPOI_rec', 'POI_rec', 'nonPOI_f', 'POI_f', 'nonPOI_sup',
                    'POI_sup', 't_neg', 'f_neg', 'f_pos', 't_pos', 'train_t', 'predict_t', 'model']
base_perf_df = pd.DataFrame(columns=ordered_cols_lst)

clf_dict = {'dt_clf': DecisionTreeClassifier, 'rf_clf': RandomForestClassifier, 'ab_clf': AdaBoostClassifier,
            'kn_clf': KNeighborsClassifier, 'gnb_clf': GaussianNB, 'svc_clf': svm.SVC}

print('\nBaseline model performance metrics, no engineered features or tuning, imputed NaNs with 0, using a train-test split of the training set:\n')
for key, method in clf_dict.items():
    _, _, _, _, perf_sr = run_skl(method=method, X_train=X_train_base_imp0,
                                  y_train=y_train_1d_base,
                                  X_test=X_test_base_imp0,
                                  y_test=y_test_1d_base,
                                  perf_series=key)
    base_perf_df = base_perf_df.append(perf_sr)
    
### Save a full copy of basic split sets for baseline comparison once final model is built.
X_train_base = X_train.copy()
X_train_base = pd.DataFrame(data=imp_0.transform(X=X_train_base), columns=X_train_base.columns,
                            index=X_train_base.index)
X_test_base = X_train.copy()
X_test_base = pd.DataFrame(data=imp_0.transform(X=X_test_base), columns=X_test_base.columns,
                           index=X_test_base.index)
y_train_base = np.ravel(y_train.astype(bool))
y_test_base = np.ravel(y_test.astype(bool))


##########################################################################################
### Task 2: Remove/handle outliers
print('Handling outliers.')
### Dropped ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'] row upon loading.

### Drop features that are too sparse.
### Drop 'other' because it's ill-defined and seems overly represented within important features.
### The nebulous nature of it seems like a good fit for fraud, but high gross 'other' amounts are more associated
### with nonPOIs than POIs if anything.
drop_feats_lst = ['director_fees', 'restricted_stock_deferred', 'other']

X_train.drop(columns=drop_feats_lst, inplace=True)
X_test.drop(columns=drop_feats_lst, inplace=True)
data_df.drop(columns=drop_feats_lst, inplace=True)

fin_features = [feat for feat in fin_features if feat not in drop_feats_lst]
pay_features = [feat for feat in pay_features if feat not in drop_feats_lst]
stock_features = [feat for feat in stock_features if feat not in drop_feats_lst]
email_features = [feat for feat in email_features if feat not in drop_feats_lst]
features_list = [feat for feat in features_list if feat not in drop_feats_lst]
del drop_feats_lst

### Don't drop records now because it will mess up the split for Udacity.
### Could drop earlier and resplit, but I've already done a lot of EDA behind the scenes.
### NaN his financials instead.
X_train.loc[['POWERS WILLIAM'], pay_features] = np.nan
data_df.loc[['POWERS WILLIAM'], pay_features] = np.nan

### Bivariate linear regression of the ratios between to/from/shared with POIs and
### total to and from messages revealed that top coding to_messages and from_messages
### may slightly aid nonPOI precision.
### Only top coding the training set in order to bias the model,
### because I am less concerned with accuracy than I am with POI recall,
### and by extension, nonPOI precision.
X_train['to_messages'] = X_train['to_messages'].apply(lambda x: x if x < 12000 or np.isnan(x) else 12000)
X_train['from_messages'] = X_train['from_messages'].apply(lambda x: x if x < 8000 or np.isnan(x) else 8000)
data_df.loc[X_train.index]['to_messages'] \
    = data_df.loc[X_train.index]['to_messages'].apply(lambda x: x if x < 12000 or np.isnan(x) else 12000)
data_df.loc[X_train.index]['from_messages'] \
    = data_df.loc[X_train.index]['from_messages'].apply(lambda x: x if x < 8000 or np.isnan(x) else 8000)

### Not sure whether top coding these will really help or hinder, if anything at all.
### But, it appears to potentially aid POI recall in some cases
### when comparing payments to totals, and it's more in line with best practices.
### Only really affects Frevert.
top = X_train['total_payments'].dropna().sort_values()[-2]
X_train['total_payments'] = X_train['total_payments'].apply(lambda x : x if x < top or np.isnan(x) else top)
data_df.loc[X_train.index]['total_payments'] \
    = data_df.loc[X_train.index]['total_payments'].apply(lambda x : x if x < top or np.isnan(x) else top)

top = X_train['long_term_incentive'].dropna().sort_values()[-2]
X_train['long_term_incentive'] = \
    X_train['long_term_incentive'].apply(lambda x : x if x < top or np.isnan(x) else top)
data_df.loc[X_train.index]['long_term_incentive'] \
    = data_df.loc[X_train.index]['long_term_incentive'].apply(lambda x : x if x < top or np.isnan(x) else top)

### Same story as Powers, NaN all of Belfer instead of simply dropping.
X_train.loc['BELFER ROBERT'] = np.nan
# belfers_poi = data_df.loc['BELFER ROBERT']['poi']
data_df.loc['BELFER ROBERT', features_list[1:]]= np.nan
# data_df.loc['BELFER ROBERT']['poi'] = belfers_poi

### After look at distributions of ratios of features, more top/bottom coding. ###

### Nan Bannantine's salary, and bottom code salary.
X_train.loc['BANNANTINE JAMES M', 'salary'] = np.nan
data_df.loc['BANNANTINE JAMES M', 'salary'] = np.nan
bottom = X_train['salary'].dropna().sort_values(ascending=False)[-2]
X_train['salary'] = X_train['salary'].apply(lambda x : x if x > bottom or np.isnan(x) else bottom)
data_df.loc[X_train.index]['salary'] \
    = data_df.loc[X_train.index]['salary'].apply(lambda x : x if x > bottom or np.isnan(x) else bottom)

### These two only have one, very low payment value.
# X_train.loc[['HAYES ROBERT E', 'HAUG DAVID L'], pay_features] = np.nan
# data_df.loc[['HAYES ROBERT E', 'HAUG DAVID L'], pay_features] = np.nan
X_train.loc[['HAYES ROBERT E'], pay_features] = np.nan
data_df.loc[['HAYES ROBERT E'], pay_features] = np.nan

### Top code deferred_income.
top = X_train['deferred_income'].dropna().sort_values(ascending=True)[-3]
X_train['deferred_income'] = X_train['deferred_income'].apply(lambda x : x if x < top or np.isnan(x) else top)
data_df.loc[X_train.index]['deferred_income'] = \
    data_df.loc[X_train.index]['deferred_income'].apply(lambda x : x if x < top or np.isnan(x) else top)
del top
del bottom


##########################################################################################
### Task 3: Create new feature(s)
print('Engineering features.')
### Start with all ratios, within respective subspaces (fin:fin, e:e).

### Add financial ratios within subspaces to data sets.
pay_feats_divby_df = set_all_ratios(df=X_train, denoms=pay_features, numers=pay_features)
stock_feats_divby_df = set_all_ratios(df=X_train, denoms=stock_features, numers=stock_features)
### Only plausible email ratios (all reciprocals still, to get the 0s to infs):
to_lst = ['to_messages', 'from_poi_to_this_person', 'shared_receipt_with_poi']
from_lst = ['from_messages', 'from_this_person_to_poi']
email_to_divby_df = set_all_ratios(df=X_train, denoms=to_lst, numers=to_lst)
email_from_divby_df = set_all_ratios(df=X_train, denoms=from_lst, numers=from_lst)

X_train = pd.concat(objs=[X_train, pay_feats_divby_df, stock_feats_divby_df, email_to_divby_df,
                          email_from_divby_df], axis=1)

### Do for test set.
pay_feats_divby_df = set_all_ratios(df=X_test, denoms=pay_features, numers=pay_features)
stock_feats_divby_df = set_all_ratios(df=X_test, denoms=stock_features, numers=stock_features)
email_to_divby_df = set_all_ratios(df=X_test, denoms=to_lst, numers=to_lst)
email_from_divby_df = set_all_ratios(df=X_test, denoms=from_lst, numers=from_lst)
X_test = pd.concat(objs=[X_test, pay_feats_divby_df, stock_feats_divby_df, email_to_divby_df,
                         email_from_divby_df], axis=1)

### Do for full set.
pay_feats_divby_df = set_all_ratios(df=data_df, denoms=pay_features, numers=pay_features)
stock_feats_divby_df = set_all_ratios(df=data_df, denoms=stock_features, numers=stock_features)
email_to_divby_df = set_all_ratios(df=data_df, denoms=to_lst, numers=to_lst)
email_from_divby_df = set_all_ratios(df=data_df, denoms=from_lst, numers=from_lst)
data_df = pd.concat(objs=[data_df, pay_feats_divby_df, stock_feats_divby_df, email_to_divby_df,
                          email_from_divby_df], axis=1)
del to_lst
del from_lst

### Set all np.inf to np.nan.
set_inf = lambda col: col.apply(func=(lambda x: np.nan if abs(x) == abs(np.inf) else x))
X_train = X_train.apply(func=set_inf)
X_test = X_test.apply(func=set_inf)
data_df = data_df.apply(func=set_inf)

### Remove all features containing less than 30% training observations.
drop_lst = list(X_train.count().loc[X_train.count() < .3 * len(X_train.index)].index)
X_train.drop(columns=drop_lst, inplace=True)
X_test.drop(columns=drop_lst, inplace=True)
data_df.drop(columns=drop_lst, inplace=True)

pay_feats_divby_lst = [feat for feat in list(pay_feats_divby_df.columns) if not feat in drop_lst]
stock_feats_divby_lst = [feat for feat in list(stock_feats_divby_df.columns) if not feat in drop_lst]
email_feats_divby_lst = [feat for feat in list(email_to_divby_df.columns) if not feat in drop_lst] \
                        + [feat for feat in list(email_from_divby_df.columns) if not feat in drop_lst]
fin_features = [feat for feat in fin_features if feat not in drop_lst] + pay_feats_divby_lst \
    + stock_feats_divby_lst
pay_features = [feat for feat in pay_features if feat not in drop_lst]
stock_features = [feat for feat in stock_features if feat not in drop_lst]
email_features = [feat for feat in email_features if feat not in drop_lst] + email_feats_divby_lst
features_list = [feat for feat in features_list if feat not in drop_lst] + pay_feats_divby_lst \
    + stock_feats_divby_lst + email_feats_divby_lst
del drop_lst


# ### Create features that flag mambership in various quantiles, outliership, and x > 0.
# ### Use multiple quantiles: quartiles, quintiles, and deciles.
# ### Retain np.nans.

# to_flag_lst = fin_features + email_features

# ### Could write a function, but I'll just paste and edit.
# ### Flag train set.
# fin_quant_flags_df = quant_flag_all(df=X_train[fin_features], quant_df=X_train[fin_features])
# email_quant_flags_df = quant_flag_all(df=X_train[email_features], quant_df=X_train[email_features])
# fin_out_flags_df = out_flag_all(df=X_train[fin_features], quant_df=X_train[fin_features])
# email_out_flags_df = out_flag_all(df=X_train[email_features], quant_df=X_train[email_features])
# sign_flags_df = flag_signs(df=X_train[to_flag_lst])
# X_train = pd.concat(objs=[X_train, fin_quant_flags_df, email_quant_flags_df, fin_out_flags_df,
#                           email_out_flags_df, sign_flags_df], axis=1)

# ### Flag test set.
# fin_quant_flags_df = quant_flag_all(df=X_test[fin_features], quant_df=X_train[fin_features])
# email_quant_flags_df = quant_flag_all(df=X_test[email_features], quant_df=X_train[email_features])
# fin_out_flags_df = out_flag_all(df=X_test[fin_features], quant_df=X_train[fin_features])
# email_out_flags_df = out_flag_all(df=X_test[email_features], quant_df=X_train[email_features])
# sign_flags_df = flag_signs(df=X_test[to_flag_lst])
# X_test = pd.concat(objs=[X_test, fin_quant_flags_df, email_quant_flags_df, fin_out_flags_df,
#                           email_out_flags_df, sign_flags_df], axis=1)

# ### Flag whole set.
# fin_quant_flags_df = quant_flag_all(df=data_df[fin_features], quant_df=X_train[fin_features])
# email_quant_flags_df = quant_flag_all(df=data_df[email_features], quant_df=X_train[email_features])
# fin_out_flags_df = out_flag_all(df=data_df[fin_features], quant_df=X_train[fin_features])
# email_out_flags_df = out_flag_all(df=data_df[email_features], quant_df=X_train[email_features])
# sign_flags_df = flag_signs(df=data_df[to_flag_lst])
# data_df = pd.concat(objs=[data_df, fin_quant_flags_df, email_quant_flags_df, fin_out_flags_df,
#                           email_out_flags_df, sign_flags_df], axis=1)

# ### Create and update feature lists.
# fin_quant_flags_lst = list(fin_quant_flags_df.columns)
# email_quant_flags_lst = list(email_quant_flags_df.columns)
# quant_flags_lst = fin_quant_flags_lst + email_quant_flags_lst

# fin_out_flags_lst = list(fin_out_flags_df.columns)
# email_out_flags_lst = list(email_out_flags_df.columns)
# out_flags_lst = fin_out_flags_lst + email_out_flags_lst

# fin_features += fin_quant_flags_lst + fin_out_flags_lst
# email_features += email_quant_flags_lst + email_out_flags_lst

# sign_flags_lst = list(sign_flags_df.columns)

# features_list = features_list + quant_flags_lst + out_flags_lst + sign_flags_lst

# del to_flag_lst
# del fin_quant_flags_df
# del email_quant_flags_df
# del fin_out_flags_df
# del email_out_flags_df
# del sign_flags_df


### Scale features.
### Would be ideal to scale in the pipeline, but I initially experimented with iterative imputation
### and feeding bools into sklearn. Not worth rewriting for this project.

### Just do min-max on floats, not bools (some are objects for now because np.nan)

float_feats_lst = fin_features + email_features
# bool_feats_lst =  sign_flags_lst

scaler = MinMaxScaler()
train_floats = pd.DataFrame(data=scaler.fit_transform(X=X_train[float_feats_lst]),
                            columns=float_feats_lst, index=X_train.index)
X_train_scaled = pd.concat(objs=[train_floats], axis=1)#, X_train[bool_feats_lst]], axis=1)

test_floats = pd.DataFrame(data=scaler.transform(X=X_test[float_feats_lst]),
                           columns=float_feats_lst,index=X_test.index)
X_test_scaled = pd.concat(objs=[test_floats], axis=1)#, X_test[bool_feats_lst]], axis=1)

all_floats = pd.DataFrame(data=scaler.transform(X=data_df[float_feats_lst]),
                          columns=float_feats_lst, index=data_df.index)
data_df_scaled = pd.concat(objs=[data_df['poi'], all_floats], axis=1)#, data_df[bool_feats_lst]], axis=1)

del float_feats_lst
del scaler
del train_floats
del test_floats
del all_floats
del X_train
del X_test
del data_df


### Impute missing values:
### Financial features to 0, email features to median, and bools to mode.
### Restore bools to bool (from object because np.nan)

imp0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mod = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

### Financial features to 0.
fin_train_df = pd.DataFrame(data=imp0.fit_transform(X=X_train_scaled[fin_features]),
                        columns=fin_features, index=X_train_scaled.index)
fin_test_df = pd.DataFrame(data=imp0.transform(X=X_test_scaled[fin_features]),
                       columns=fin_features, index=X_test_scaled.index)
fin_all_df = pd.DataFrame(data=imp0.transform(X=data_df_scaled[fin_features]),
                      columns=fin_features, index=data_df_scaled.index)

### email features to median.
email_train_df = pd.DataFrame(data=imp_med.fit_transform(X=X_train_scaled[email_features]),
                        columns=email_features, index=X_train_scaled.index)
email_test_df = pd.DataFrame(data=imp_med.transform(X=X_test_scaled[email_features]),
                       columns=email_features, index=X_test_scaled.index)
email_all_df = pd.DataFrame(data=imp_med.transform(X=data_df_scaled[email_features]),
                      columns=email_features, index=data_df_scaled.index)

# ### Bools to mode.
# ### Restore bools to bool (from object because np.nan)
# bool_train_df = (pd.DataFrame(data=imp_mod.fit_transform(X=X_train_scaled[bool_feats_lst]),
#                               columns=bool_feats_lst, index=X_train_scaled.index)).astype(bool)
# bool_test_df = pd.DataFrame(data=imp_mod.transform(X=X_test_scaled[bool_feats_lst]),
#                             columns=bool_feats_lst, index=X_test_scaled.index).astype(bool)
# bool_all_df = pd.DataFrame(data=imp_mod.transform(X=data_df_scaled[bool_feats_lst]),
#                            columns=bool_feats_lst, index=data_df_scaled.index).astype(bool)

### Concat
X_train_scaled_imp = pd.concat(objs=[fin_train_df, email_train_df], axis=1)#, bool_train_df], axis=1)
X_test_scaled_imp = pd.concat(objs=[fin_test_df, email_test_df], axis=1)#, bool_test_df], axis=1)
data_df_scaled_imp = pd.concat(objs=[data_df_scaled['poi'], fin_all_df, email_all_df], axis=1)#, bool_all_df], axis=1)

del fin_train_df
del email_train_df
# del bool_train_df
del fin_test_df
del email_test_df
# del bool_test_df
del fin_all_df
del email_all_df
# del bool_all_df
# del bool_feats_lst
del X_train_scaled
del X_test_scaled
del data_df_scaled


# ### sklearn predictions as features

# # 1) Kmeans cluster.
# train_cluster_subspace, test_cluster_subspace \
#     = add_k_means_n(X_train=X_train_scaled_imp, X_test=X_test_scaled_imp)
# X_train_scaled_imp_k = pd.concat(objs=[X_train_scaled_imp, train_cluster_subspace], axis=1)
# X_test_scaled_imp_k = pd.concat(objs=[X_test_scaled_imp, test_cluster_subspace], axis=1)

# train_cluster_subspace, test_cluster_subspace \
#     = add_k_means_n(X_train=X_train_scaled_imp, X_test=data_df_scaled_imp[features_list[1:]])
# data_df_scaled_imp_k = pd.concat(objs=[data_df_scaled_imp, test_cluster_subspace], axis=1)

# k_means_feats_lst = k_means_feats_lst = list(train_cluster_subspace.columns)
# features_list += k_means_feats_lst

# del train_cluster_subspace
# del test_cluster_subspace
# del X_train_scaled_imp
# del X_test_scaled_imp
# del data_df_scaled_imp


################################################################################################
### [Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html]

# # [Provided to give you a starting point. Try a variety of classifiers.]
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### Construct baseline performance with all features before tuning/selection.
print('\nBaseline models using all engineered features, mixed imputation, train-test split of train set:\n')
### Split train set again for a baseline model that won't touch the final test set.
X_train_base, X_test_base, y_train_base, y_test_base \
    = train_test_split(X_train_scaled_imp, y_train, test_size=.3, random_state=42)
#     = train_test_split(X_train_scaled_imp_k, y_train, test_size=.3, random_state=42)
y_train_1d_base = np.ravel(y_train_base.astype(bool))
y_test_1d_base = np.ravel(y_test_base.astype(bool))

### Save metrics as a dataframe if you want to save the object and inspect it later.
base_perf_engineered_df = pd.DataFrame(columns=ordered_cols_lst)

base_perfs_dict = {'base_perf_engineered': base_perf_engineered_df}
imp_sets_dict = {'base_perf_engineered': [X_train_base, X_test_base]}

### Modifies the base_perfs_dict in place, because dict has no deep copy method.
get_base_perfs(base_perfs_dict=base_perfs_dict, imp_sets_dict=imp_sets_dict, clf_dict=clf_dict, y_train=y_train_1d_base,
               y_test=y_test_1d_base)

base_perfs_dict['first_base'] = base_perf_df


################################################################################################
### [Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html]

### Because the proliferation of features has led to overfit
### (see gridsearch notebooks in the supplemental material folder),
### remove quantile flags, outlier flags, sign flags, and cluster flags,
### leaving the original base features (that were not removed) and the ratio features.

drop_lst = []#quant_flags_lst + out_flags_lst + sign_flags_lst + k_means_feats_lst
keep_lst = [feat for feat in features_list[1:] if feat not in drop_lst]

# X_train_trimmed = X_train_scaled_imp_k[keep_lst]
# X_test_trimmed = X_test_scaled_imp_k[keep_lst]
# data_df_trimmed = data_df_scaled_imp_k[['poi'] + keep_lst]
X_train_trimmed = X_train_scaled_imp[keep_lst]
X_test_trimmed = X_test_scaled_imp[keep_lst]
data_df_trimmed = data_df_scaled_imp[['poi'] + keep_lst]

print('Tuning model.')
### GridSearchCV inputs:
n_jobs = -1
### Callables to pass into parameter grid:
mutual_info_classif_partial = partial(mutual_info_classif, random_state=42)
DecisionTreeClassifier_partial = partial(DecisionTreeClassifier, random_state=42)
RandomForestClassifier_partial = partial(RandomForestClassifier, random_state=42, n_jobs=n_jobs)
AdaBoostClassifier_partial = partial(AdaBoostClassifier, random_state=42)
svm_SVC_partial = partial(svm.SVC, random_state=42)
KNeighborsClassifier_partial = partial(KNeighborsClassifier, n_jobs=n_jobs)

### Would be ideal to scale in the pipeline, but I initially experimented with iterative imputation
### and feeding bools into sklearn, and so scaled first. Not worth rewriting for this project.

selectors = {
    'sel_per': {
        'sel': SelectPercentile(),
        'params': {
            'sel_per__score_func': [f_classif, chi2, mutual_info_classif_partial],
            'sel_per__percentile': [2, 4, 6, 8, 10, 12, 14]
        }
    }
}

decomps = {
    'empty' : None
#     'fica': {
#         'dec': FastICA(),
#         'params': {
#             'fica__algorithm': ['parallel', 'deflation'],
#             'fica__fun': ['logcosh', 'exp', 'cube'],
#             'fica__random_state': [42]
#         }
#     },
#         'ipca': {
#         'dec': IncrementalPCA(),
#         'params': {
#             ### defaults
#         }
#     },
#     'kpca': {
#         'dec': KernelPCA(),
#         'params': {
#             'kpca__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine',
#                              'precomputed'],
#             'kpca__random_state': [42],
#             'kpca__n_jobs': [n_jobs]
#         }
#     },
    ### PCA kept throwing an error that the data contained nans, inf,
    ### or too large dtypes, despite no nans, infs, nor wrong types per
    ### replications of the sklearn (and numpy) condition checks that threw the
    ### errors ("errors" because using PCA threw the error from sklearn script,
    ### but further investigation showed that PCA's get_precision method (or another method) may have
    ### thrown it from a NumPy script (see gridsearch notebook for tracing), and that
    ### did not include dtype size).
    ### Maybe a problem with the transformed data handed off from
    ### SelectPercentile, but I'm done messing around with it. Just need to finish. Skip PCA.
#     'pca': {
#         'dec': PCA(),
#         'params': {
#             'pca__random_state': [42]
#         }
#     },
#     'tsvd': {
#         'dec': TruncatedSVD(),
#         'params': {
#             'tsvd__n_components': [2, 4, 8, 16, 32, 64, 128],
#             'tsvd__algorithm': ['arpack', 'randomized'],
#             'tsvd__random_state': [42]
#         }
#     }
}

classifiers = {
#     'dt_clf': {
#         'clf': DecisionTreeClassifier(),
#         'params': {
#             'dt_clf__random_state': [42]
#         }
#     },
    'rf_clf': {
        'clf': RandomForestClassifier(),
        'params': {
            'rf_clf__n_estimators': [1, 2, 3, 4, 5, 6, 7],
            'rf_clf__max_features': ['sqrt', 'log2'],
            'rf_clf__max_depth': [8, 16, 24],
            'rf_clf__min_samples_split': [2],
            'rf_clf__min_samples_leaf': [1, 2, 3, 4],
            'rf_clf__bootstrap': [True, False],
            'rf_clf__random_state': [42],
            'rf_clf__n_jobs': [n_jobs]
        }
    },
#     'ab_clf': {
#         'clf': AdaBoostClassifier(),
#         'params': {
#             'ab_clf__base_estimator': [
#                 DecisionTreeClassifier_partial(),
#                 RandomForestClassifier_partial(),
#                 AdaBoostClassifier_partial(),
#                 svm_SVC_partial(),
#                 KNeighborsClassifier_partial(),
#                 GaussianNB()
#             ],
#             'ab_clf__n_estimators': [8, 16, 24, 32, 40, 48, 56],
#             'ab_clf__algorithm': ['SAMME', 'SAMME.R'],
#             'ab_clf__random_state': [42]
#         }
#     },
#     'kn_clf': {
#         'clf': KNeighborsClassifier(),
#         'params': {
#             'kn_clf__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#             'kn_clf__weights': ['uniform', 'distance'],
#             'kn_clf__algorithm': ['ball_tree', 'kd_tree', 'brute'],
#             'kn_clf__leaf_size': [4, 8, 12, 16, 20, 24, 30],
#             'kn_clf__n_jobs': [n_jobs]
#         }
#     },
#     'gnb_clf': {
#         'clf': GaussianNB(),
#         'params': {
#             # Defaults
#         }
#     },
}


print('\nGrid search output:\n')
imp_gscvs_dict = {}
imp_gscvs_dict['mixed_impute_trimmed'] \
    = search_em_all(X_train=X_train_trimmed, y_train=y_train_1d, selectors=selectors,
                    decomps=decomps, classifiers=classifiers, pipe_verbose=True,
                    scoring='recall_weighted', n_jobs=-1)
### Can try with multiple datasets for comparison.
# imp_gscvs_dict['other_set'] \
#     = search_em_all(X_train=X_train_other_set, y_train=y_train_1d, selectors=selectors,
#                     decomps=decomps, classifiers=classifiers, pipe_verbose=True,
#                     scoring='recall_weighted', n_jobs=-1)

with open('data/imp_gscvs_dict_last.pkl', 'wb') as file:
    pickle.dump(obj=imp_gscvs_dict, file=file)

    
    
### Check some performance metrics of final model.
get_f = lambda precision, recall: 2 * ((precision * recall) / (precision + recall))
for name, gscv in imp_gscvs_dict['mixed_impute_trimmed'].items():
    print('Best model key:', name, '\n')
    print('Best score:\n', gscv.best_score_, '\n')
    print('Best estimator:\n', gscv.best_estimator_, '\n')
    clf = gscv.best_estimator_
    pred = clf.predict(X_test_trimmed)
    conf = confusion_matrix(y_true=y_test_1d, y_pred=pred)
    print('Confusion matrix:\n', conf, '\n')
    prf = precision_recall_fscore_support(y_true=y_test_1d, y_pred=pred)
    print('Precision, recall, f beta score, support:\n', prf, '\n')
    print('Custom F beta using nonPOI precision and POI recall:\n', get_f(prf[0][0], prf[1][1]), '\n')
    print('\n')
    
    
# ### Compare to baseline using whole set.
# ### Store metrics in a dataframe in case you want to save and inspect it.
# base_perf_full_df= pd.DataFrame(columns=ordered_cols_lst)
# print('\nBaseline models, no engineered features, no tuning, impute with 0, full train-test split:\n')
# for key, method in clf_dict.items():
#     _, _, _, prf, perf_sr = run_skl(method=method, X_train=X_train_base,
#                                   y_train=y_train_base,
#                                   X_test=X_test_base,
#                                   y_test=y_test_base,
#                                   perf_series=key)
#     print('Custom F beta using nonPOI precision and POI recall:\n', get_f(prf[0][0], prf[1][1]), '\n')
#     base_perf_full_df = base_perf_full_df.append(perf_sr)
    
# print('\nBaseline models, cleaned set, human-selected and engineered features, scaled, mixed imputation, no tuning:\n')
# base_perf_engineered_trimmed_df = pd.DataFrame(columns=ordered_cols_lst)
# for key, method in clf_dict.items():
#     _, _, _, prf, perf_sr = run_skl(method=method, X_train=X_train_trimmed,
#                                   y_train=y_train_1d,
#                                   X_test=X_test_trimmed,
#                                   y_test=y_test_1d,
#                                   perf_series=key)
#     print('Custom F beta using nonPOI precision and POI recall:\n', get_f(prf[0][0], prf[1][1]), '\n')
#     base_perf_full_df = base_perf_df.append(perf_sr)
    
    
### Check final features.
print('\nAll features and their scores provided by scoring function mutual_info_classif:')
feature_scores_sr = pd.Series(data=clf.named_steps['sel_per'].scores_, index=X_train_trimmed.columns)
print(feature_scores_sr.sort_values(ascending=False))

print('\nSelected features and their scores provided by selecting function mutual_info_classif:')
num_selected = int((0.01 * clf.named_steps['sel_per'].percentile) * len(X_train_trimmed.columns))
print(feature_scores_sr.sort_values(ascending=False).head(num_selected))


### [Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results ...]
print('Saving classifier, data, and features list.')
features_list = keep_lst
data_df_trimmed['poi'] = data_df_trimmed['poi'].astype(bool)
my_dataset = data_df_trimmed.T.to_dict()
### Would be ideal to scale in the pipeline, but I initially experimented with iterative imputation
### and feeding bools into sklearn, and so scaled first. Not worth rewriting for this project.
clf = imp_gscvs_dict['mixed_impute_trimmed']['sel_per_empty_rf_clf'].best_estimator_

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

with open(CLF_PICKLE_FILENAME, "wb") as clf_outfile:
    pickle.dump(clf, clf_outfile)
with open(DATASET_PICKLE_FILENAME, "wb") as dataset_outfile:
    pickle.dump(my_dataset, dataset_outfile)
with open(FEATURE_LIST_FILENAME, "wb") as featurelist_outfile:
    pickle.dump(features_list, featurelist_outfile)
    
### [... You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.]

### Deprecated
# dump_classifier_and_data(clf, my_dataset, features_list)