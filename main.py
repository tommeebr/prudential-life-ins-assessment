from libraries import pd, np, xgb, skl, plt, lgb, optuna, ft, XGBOrdinal
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


# Training the model using XGB
# Softmax function turns scores into probabilities and then picks the class with the highest probability
# (The most likely response)
model= xgb.XGBClassifier(objective='multi:softmax', num_class=8, random_state=12)
model.fit(X_train, y_train)


# Calculate and print the Quadratic Weighted Kappa (QWK) score.
# QWK measures how well the predicted classes agree with the true classes,
# giving more penalty for predictions that are further away from the true value.
# QWK ranges from -1 (complete disagreement) to 1 (perfect agreement), with 0 meaning random agreement.
y_pred = model.predict(X_val)
print("xgb Quadratic Weighted Kappa:", skl.metrics.cohen_kappa_score(y_val, y_pred, weights='quadratic'))



# Testing xgbordinal. The problem with xgboost was the fact it wouldn't respond to the ordinal nature of response.
# The "Response" column is ordinal because its values (1–8) have a natural order—higher numbers mean higher risk.
# Standard XGBoost treats these as just separate categories, ignoring the order between them.
# XGBOrdinal is designed for ordinal data, so it can better capture the ordered relationship between response levels.

ordinal_model = XGBOrdinal(
    aggregation='weighted', 
    norm=True,
    random_state=23
)

ordinal_model.fit(X_train, y_train)

y_pred_ordinal = ordinal_model.predict(X_val)

print("XGBOrdinal Quadratic Weighted Kappa:", skl.metrics.cohen_kappa_score(y_val, y_pred_ordinal, weights='quadratic'))



# * Hyperparameter Tuning!

# Custom scorer (uses qwk for accuracy mesasurements)
qwk_scorer = skl.metrics.make_scorer(skl.metrics.cohen_kappa_score, weights='quadratic')

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 250, 500]
}

ordinal_model = XGBOrdinal(
    aggregation='weighted',
    norm=True,
    random_state=23
)

grid = skl.model_selection.GridSearchCV(
    estimator=ordinal_model,
    param_grid=param_grid,
    scoring=qwk_scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters found:", grid.best_params_)
print("Best cross-validation QWK:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_val)
print("XGBOrdinal QWK (tuned):", skl.metrics.cohen_kappa_score(y_val, y_pred, weights='quadratic'))
