import os

import math
import numpy as np
import pandas as pd
import sklearn

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

import pickle

import logging
logging.basicConfig(filename='./logs/train.log', level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def std_col_names(df):
    """
    - Convert feature names to lower case
    - Rename reporting date column
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df.rename(columns = {'employment_duration': 'property', 'debit_to_income': 'debt_to_income', 'home_ownership': 'emp_duration'}, inplace= True)
    df['loan_title'] = df['loan_title'].str.lower().str.strip().str.replace(' ', '_')
    return df

# def process_data(df, dv)

def transform_dv(df, dv=None, train= False):
    """
    DictVectorize a dataframe
    """
    df_dict = df.to_dict(orient= 'records')

    if train:
        dv = DictVectorizer(sparse= False)
        train_dv = dv.fit_transform(df_dict)
        return df_dict, dv, train_dv
    else:
        try:
            val_dv = dv.transform(df_dict)
            return df_dict, val_dv
        except TypeError:
            print('DictVectorizer was not passed for non-train set')

def main():
    """
    Driver utility for training model
    """
    # feature list after all processing is performed
    # the model will be trained on these feature columns
    skim_num_cols = [
    'revolving_utilities',
    'total_received_interest',
    'interest_rate',
    'debt_to_income',
    'funded_amount_investor',
    'total_received_late_fee',
    'recoveries',
    'collection_recovery_fee',
    'total_current_balance',
    'total_revolving_credit_limit',
    'loan_amount',
    'funded_amount',
    'revolving_balance',
    'bal_acc'
    ]

    cat_cols =[
    'grade',
    'property',
    'verification_status',
    'delinquency_-_two_years',
    'collection_12_months_medical',
    'application_type',
    'initial_list_status',
    ]

    df = std_col_names(pd.read_csv('./data/train.csv'))
    logging.info('Dataset loaded')
    y_count = df['loan_status'].value_counts()
    logging.info(f'Dataset label value counts:\n{y_count}')

    df['bal_acc'] = df['total_accounts']-df['open_account']
    train = df.copy()
    logging.info(f'Sample: ')
    logging.info(train.columns)

    # train_dict = train.to_dict(orient= 'records')
    # logging.info('Data processed into dictionary')

    # instantiate DictVectorizer
    # dv = DictVectorizer(sparse= False)
    # train_dv = dv.fit_transform(train_dict)
    # logging.info('Data transformed using DictVectorizer')
    train_dict, dv, train_dv = transform_dv(train[skim_num_cols+cat_cols], train= True)

    over = SMOTE(sampling_strategy=0.25)
    x_over, y_over = over.fit_resample(train_dv, df['loan_status'])

    under = RandomUnderSampler(sampling_strategy= 0.5)
    x_ou, y_ou = under.fit_resample(x_over, y_over)
    y_ou_count = pd.Series(y_ou).value_counts()
    logging.info(f'Dataset label value counts:\n{y_ou_count}')


    # instantiate model
    fin_params = {
    'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1,
      'colsample_bynode':1, 'colsample_bytree':1, 'enable_categorical':False,
      'eval_metric':'logloss', 'gamma':0,  'importance_type':None,
      'interaction_constraints':'', 'learning_rate':0.15, 'max_delta_step':2,
      'max_depth':5, 'min_child_weight':4,
      'monotone_constraints':'()', 'n_estimators':50, 'n_jobs':8,
      'num_parallel_tree':1, 'predictor':'auto', 'random_state':0,
      'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'subsample':0.8,
      'use_label_encoder':False,
      'validate_parameters':1, 'verbosity':2 #'gpu_id':-1,'tree_method':'gpu_hist',
    }

    dtrain = xgb.DMatrix(x_ou, y_ou)
    # rf_tuned = RandomForestRegressor(ccp_alpha=0.05, max_depth=15, max_features=5,
    #                   max_samples=0.5, n_estimators=200)
    logging.info('Training Model')

    # fit model to transformed train data
    fin_xgb = xgb.train(fin_params, dtrain, num_boost_round = 200)
    logging.info('Model trained!')

    # export data objects required for making predictions
    model_file = 'loan_model.bin'
    dv_file = 'loan_dv.bin'

    with open(model_file, 'wb') as outfile:
        pickle.dump(fin_xgb, outfile)

    with open(dv_file, 'wb') as outfile:
        pickle.dump(dv, outfile)

    logging.info(f'Objects exported!: {model_file}, {dv_file}')
    print('Training completed. Check logs at ./logs/train.log')


if __name__ == "__main__":
    main()
