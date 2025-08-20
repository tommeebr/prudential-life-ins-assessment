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




# Train Gradient Boosting model
gb_model = skl.ensemble.GradientBoostingClassifier(
    n_estimators=250,      
    max_depth=5,          
    learning_rate=0.1,    
    random_state=23
)
gb_model.fit(X_train, y_train)

# Predict and evaluate
gb_y_pred = gb_model.predict(X_val)
print("GradientBoosting QWK:", skl.metrics.cohen_kappa_score(y_val, gb_y_pred, weights='quadratic'))