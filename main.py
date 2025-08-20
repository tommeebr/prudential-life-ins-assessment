from libraries import pd, np, xgb, skl, plt, lgb, optuna, ft
# Reading csv with pd to create DataFrames
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

missing_train = df_train.isnull().sum()
missing_test = df_test.isnull().sum()

#? print('Missing values in train:')
#? print(missing_train[missing_train > 0])
#? print('\nMissing values in test:')
#? print(missing_test[missing_test > 0])

# Identify numeric and categorical columns
num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

#? print('\nNumeric columns:', num_cols)
#? print('Categorical columns:', cat_cols)

# Basic imputation for missing values
for col in num_cols:
    if df_train[col].isnull().any():
        median = df_train[col].median()
        df_train[col] = df_train[col].fillna(median)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(median)
for col in cat_cols:
    if df_train[col].isnull().any():
        mode = df_train[col].mode()[0]
        df_train[col] = df_train[col].fillna(mode)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(mode)

# Check missing values after imputation
#? print('\nMissing values after imputation (train):')
#? print(df_train.isnull().sum().sum())



# Label Encoding for categorical columns
le = skl.preprocessing.LabelEncoder()

# Before encoding (test)
#? print("Unique values in Product_Info_2 before encoding:", df_train['Product_Info_2'].unique())

for col in cat_cols:
    # Fit on combined data to ensure consistency
    le.fit(list(df_train[col]) + list(df_test[col]))
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# After encoding (test)
#? print("Unique values in Product_Info_2 after encoding:", df_train['Product_Info_2'].unique())



#* Features (inputs)
X = df_train.drop('Response', axis=1)

#* Target (-1 so values start at 0)
y = df_train['Response'] - 1

# Setting up training and validation sets
# We split data into 80% training set and 20% validation set
# This is done to evaluate the models performance on unseen data
X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y, test_size=0.2, random_state=23)

'''
# Training the model using XGB
# Softmax function turns scores into probabilities and then picks the class with the highest probability
# (The most likely response)
model= xgb.XGBClassifier(objective='multi:softmax', num_class=8, random_state=12)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("xgb Validation accuracy:", skl.metrics.accuracy_score(y_val, y_pred))

# Calculate and print the Quadratic Weighted Kappa (QWK) score.
# QWK measures how well the predicted classes agree with the true classes,
# giving more penalty for predictions that are further away from the true value.
# QWK ranges from -1 (complete disagreement) to 1 (perfect agreement), with 0 meaning random agreement.
print("xgb Quadratic Weighted Kappa:", skl.metrics.cohen_kappa_score(y_val, y_pred, weights='quadratic'))
'''

# Optuna was used to optimize the hyperparameters, but tuning didn't help much.

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=8,
    random_state=23,
    verbose=-1
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
print("lgb Validation accuracy:", skl.metrics.accuracy_score(y_val, y_pred))
print("lgb Quadratic Weighted Kappa:", skl.metrics.cohen_kappa_score(y_val, y_pred, weights='quadratic'))



# ** Experimenting with feature engineering

# Total number of medical keywords flagged for each applicant
df_train['Medical_Keyword_Sum'] = df_train[[col for col in df_train.columns if 'Medical_Keyword' in col]].sum(axis=1)
df_test['Medical_Keyword_Sum'] = df_test[[col for col in df_test.columns if 'Medical_Keyword' in col]].sum(axis=1)

# BMI x Age: Could relate to health risk
df_train['BMI_Age'] = df_train['BMI'] * df_train['Ins_Age']
df_test['BMI_Age'] = df_test['BMI'] * df_test['Ins_Age']

# Sum of all family history columns
fam_cols = [col for col in df_train.columns if 'Family_Hist' in col]
df_train['Family_Hist_Sum'] = df_train[fam_cols].sum(axis=1)
df_test['Family_Hist_Sum'] = df_test[fam_cols].sum(axis=1)

# Employment info sum
emp_cols = [col for col in df_train.columns if 'Employment_Info' in col]
df_train['Employment_Info_Sum'] = df_train[emp_cols].sum(axis=1)
df_test['Employment_Info_Sum'] = df_test[emp_cols].sum(axis=1)

# Insurance History Sum
ins_cols = [col for col in df_train.columns if 'Insurance_History' in col]
df_train['Insurance_History_Sum'] = df_train[ins_cols].sum(axis=1)
df_test['Insurance_History_Sum'] = df_test[ins_cols].sum(axis=1)

# Log Transform of BMI
df_train['Log_BMI'] = np.log1p(df_train['BMI'])
df_test['Log_BMI'] = np.log1p(df_test['BMI'])

# Update features to include new engineered columns
X = df_train.drop('Response', axis=1)
y = df_train['Response'] - 1  # Keep target as before

# Split again (or reuse previous split if you want)
X_train, X_val, y_train, y_val = skl.model_selection.train_test_split(X, y, test_size=0.2, random_state=23)

# Retrain the model
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=8,
    random_state=23,
    verbose=-1
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
print("lgb Validation accuracy (with new features):", skl.metrics.accuracy_score(y_val, y_pred))
print("lgb Quadratic Weighted Kappa (with new features):", skl.metrics.cohen_kappa_score(y_val, y_pred, weights='quadratic'))



# * Feature Engineering with Featuretools

X_base = df_train.drop('Response', axis=1)
y = df_train['Response'] - 1

X_train_base, X_val_base, y_train, y_val = skl.model_selection.train_test_split(X_base, y, test_size=0.2, random_state=23)

es_train = ft.EntitySet(id='prudential_train')
es_train = es_train.add_dataframe(dataframe_name='applicants', dataframe=X_train_base, index='Id')
feature_matrix_train, _ = ft.dfs(
    entityset=es_train,
    target_dataframe_name='applicants',
    max_depth=1
)

es_val = ft.EntitySet(id='prudential_val')
es_val = es_val.add_dataframe(dataframe_name='applicants', dataframe=X_val_base, index='Id')
feature_matrix_val, _ = ft.dfs(
    entityset=es_val,
    target_dataframe_name='applicants',
    max_depth=1
)

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=8,
    random_state=23,
    verbose=-1
)
model.fit(feature_matrix_train, y_train)
y_pred = model.predict(feature_matrix_val)
print("lgb Validation accuracy (with featuretools):", skl.metrics.accuracy_score(y_val, y_pred))
print("lgb Quadratic Weighted Kappa (with featuretools):", skl.metrics.cohen_kappa_score(y_val, y_pred, weights='quadratic'))
