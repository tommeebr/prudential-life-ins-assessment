from libraries import pd, np, xgb, skl, plt
#DataFrames
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
print("Unique values in Product_Info_2 before encoding:", df_train['Product_Info_2'].unique())

for col in cat_cols:
    # Fit on combined data to ensure consistency
    le.fit(list(df_train[col]) + list(df_test[col]))
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# After encoding (test)
print("Unique values in Product_Info_2 after encoding:", df_train['Product_Info_2'].unique())