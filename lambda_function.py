import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn

# load data objects required for making predictions
model_file = 'loan_model.bin'
dict_file = 'loan_dv.bin'

with open(model_file, 'rb') as infile:
    model = pickle.load(infile)

with open(dict_file, 'rb') as infile:
    dv = pickle.load(infile)



def transform_data(data):
    """
    Fetch request json
    Transform Data

    input
        data -> input received from requests

    output
        dtest -> XGB DMatrix
        df['id'] -> Series of loan applicant IDs
    """

    # model feature columns
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

    # Fetch requests made to endpoint and convert to DataFrame

    df = pd.DataFrame.from_dict(data).reset_index(drop= True)

    # Add new feature 'bal_acc'
    df['bal_acc'] =  df['total_accounts']-df['open_account']

    # Transform to DMatrix
    loan_dict = df[skim_num_cols+cat_cols].to_dict(orient= 'records')
    X = dv.transform(loan_dict)
    dtest = xgb.DMatrix(X)

    return dtest, df['id']

def predict(dmat, ids):
    """
    Makes prediction and aggregates result

    input
        dmat -> DMatrix to be input to model
        ids -> Applicant IDs for which predictions are being max_depth

    output
        result -> Dictionary with value `IDs: output::probability of default`
    """
    y_pred = model.predict(dmat)
    op = [f'Non-default:{val}' if val < 0.35 else f'Default::{round(val, 2)}' for val in y_pred]
    application_ids = [sample for sample in ids]

    result = {
        'prediction': dict(zip(application_ids, op))
    }

    return result

def lambda_handler(event, context=None):
    """
    Aggregation of operations for the lambda function and return output
    """
    loan_details = event['records']
    X, ids = transform_data(loan_details)
    op = predict(X, ids)
    return op
