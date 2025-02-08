# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
# 1. Impute Missing Values
df = pd.read_csv("C:\\Users\\sabri\\OneDrive\\Desktop\\Assignment1\\Assignment1\\Data\\messy_data.csv")
df.head()

print(df.shape)

#changes values to boolean T or F, which allows us to see missing values in the dataset
df.isnull()

df.isnull().sum()

#filling null values
df2 = df.fillna(value = 0)
print(df2)

# Filling null values with previous value
df4 = df.fillna(method = 'pad')
print(df4)

#checking for NaN in df4
df4.isnull().sum()

#Filling NaNs with mean of their column (j, n, q, u, w, z)
df9 = df.fillna(value=df['j'].mean())
print(df9)

df10 = df.fillna(value=df['n'].mean())
print(df10)

df11 = df.fillna(value=df['q'])
print(df11)

df12 = df.fillna(value=df['u'])
print(df12)

df13 = df.fillna(value=df['w'])
print(df13)

df14 = df.fillna(value=df['z'])
print(df14)

# 2. Remove Duplicates
import pandas as pd
import ast

dict = ast.literal_eval("C:\\Users\\sabri\\OneDrive\\Desktop\\Assignment1\\Assignment1\\Data\\messy_data.csv")
df_test = data(index = idx, data=(dict))


data = pd.DataFrame("C:\\Users\\sabri\\OneDrive\\Desktop\\Assignment1\\Assignment1\\Data\\messy_data.csv")

#Using df.drop_duplicated() to remove duplicate rows
data.drop_duplicates(inplace = True)

# 3. Normalize Numerical Data
import numpy as np

data = np.array("C:\\Users\\sabri\\OneDrive\\Desktop\\Assignment1\\Assignment1\\Data\\messy_data.csv")

#Using Min-Max Normalization
x_min = data.min()
x_max = data.max()
data_normalized = (data - x_min) / (x_max - x_min)

print(data_normalized)

#NOTE: I struggled a lot with my environments the last three days (utilizing PyCharm Community
#to do the rough work of this assignment before submission, but I ran out of time and was unable to fully complete the coding
#aspect of this assignment, as well as some short answers. My sincere apologies for this mishap. 

#NOTE: My environment gave me many errors that even after careful research and consideration from online sources i could not
#seem to correctly troubleshoot my errors. I will reach out to you for help regarding this situation so it does not
#continue into assignment two. Many thanks! 
# - Sabrina :) 

# 4. Remove Redundant Features - UNAVAILABLE CODE
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    pass

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None
