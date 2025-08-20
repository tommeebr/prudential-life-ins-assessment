import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return cohen_kappa_score(yhat, y, weights='quadratic')
    
def get_params():
    params = {}
    params["objective"] = "reg:squarederror"     
    params["eta"] = 0.05
    params["min_child_weight"] = 240
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.67
    params["max_depth"] = 6
    plst = list(params.items())
    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# columns to drop
columns_to_drop = ['Id', 'Response']
xgb_num_rounds = 800
num_classes = 8
eta_list = [0.05] * 200 
eta_list = eta_list + [0.02] * 500
eta_list = eta_list + [0.01] * 100

print("Load the data using pandas")
train = pd.read_csv("train.csv")  # Use your local train.csv

# create any new variables    
if 'Product_Info_2' in train.columns:
    train['Product_Info_2_char'] = train.Product_Info_2.str[0]
    train['Product_Info_2_num'] = train.Product_Info_2.str[1]
    train['Product_Info_2'] = pd.factorize(train['Product_Info_2'])[0]
    train['Product_Info_2_char'] = pd.factorize(train['Product_Info_2_char'])[0]
    train['Product_Info_2_num'] = pd.factorize(train['Product_Info_2_num'])[0]

if 'BMI' in train.columns and 'Ins_Age' in train.columns:
    train['BMI_Age'] = train['BMI'] * train['Ins_Age']

med_keyword_columns = train.columns[train.columns.str.startswith('Medical_Keyword_')]
if len(med_keyword_columns) > 0:
    train['Med_Keywords_Count'] = train[med_keyword_columns].sum(axis=1)

print('Eliminate missing values')    
train.fillna(-1, inplace=True)

# fix the dtype on the label column
train['Response'] = train['Response'].astype(int)

# split into train/validation
train_set, val_set = train_test_split(train, test_size=0.2, random_state=42, stratify=train['Response'])

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train_set.drop(columns_to_drop, axis=1), train_set['Response'].values)
xgval = xgb.DMatrix(val_set.drop(columns_to_drop, axis=1), val_set['Response'].values)    

# get the parameters for xgboost
plst = get_params()
print(plst)      

# train model
model = xgb.train(plst, xgtrain, xgb_num_rounds)

# get preds
val_preds = model.predict(xgval)
val_preds = np.clip(val_preds, -0.99, 8.99)

# train offsets 
offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
data = np.vstack((val_preds, val_preds, val_set['Response'].values))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
for j in range(num_classes):
    val_offset = lambda x: -apply_offset(data, x, j)
    offsets[j] = float(fmin_powell(val_offset, offsets[j]))

# apply offsets to validation preds
data = np.vstack((val_preds, val_preds, val_set['Response'].values))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

final_val_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

print('Validation QWK (with offsets):', cohen_kappa_score(final_val_preds, val_set['Response'].values, weights='quadratic'))