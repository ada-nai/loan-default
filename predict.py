# import packages
import pickle
from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd
import numpy as np
import xgboost as xgb

# load data objects required for making predictions
model_file = 'loan_model.bin'
dict_file = 'loan_dv.bin'

with open(model_file, 'rb') as infile:
    model = pickle.load(infile)

with open(dict_file, 'rb') as infile:
    dv = pickle.load(infile)

# create app instance
app = Flask('loan_default')

@app.route('/predict', methods= ['POST'])
def predict():
    """
    Fetch request json
    Transform Data
    Make predictions
    Return predictions as json
    """
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

    # fetch requests made to endpoint
    loan_details = request.get_json()

    df = pd.DataFrame.from_dict(loan_details).reset_index(drop= True)

    df['bal_acc'] =  df['total_accounts']-df['open_account']

    loan_dict = df[skim_num_cols+cat_cols].to_dict(orient= 'records')

    X = dv.transform(loan_dict)
    dtest = xgb.DMatrix(X)

    y_pred = model.predict(dtest)
    op = [f'Non-default:{val}' if val < 0.35 else f'Default::{round(val, 2)}' for val in y_pred]
    application_ids = [sample['id'] for sample in loan_details]

    result = {
        'prediction': dict(zip(application_ids, op))
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug= True, host= '0.0.0.0', port= 5502)
