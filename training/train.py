#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import lightgbm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor, VotingRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
from sklearn.model_selection import (
    KFold,
    RepeatedKFold
)


# In[4]:


# Split the dataframe into test and train data
def split_data(data_df):
    """Split a dataframe into training and testing datasets"""

    x = data_df.drop(['car1_prem_mean'], axis=1) # all columns except the target column
    y = data_df['car1_prem_mean']                # only the target column
    
    # split the dataset: 70% for train set and 30% for test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 
    
    train_data = x_train, y_train 
    valid_data = x_test, y_test 
    #train_data = lightgbm.Dataset(x_train, label=y_train)  # create the train dataset
    #valid_data = lightgbm.Dataset(x_test, label=y_test, free_raw_data=False) # create the test dataset

    return (train_data, valid_data)


# In[4]:


# Test if the dataset contains null values
# The isnull() function will count the number of nulls that exist on the dataset
# The first sum() will return a Series with the count of null values in each column
# The second sum() returns the total count of null values in the entire DataFrame.
def test_null_values(df):
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"There are {null_count} null values in the DataFrame"

# Test if the dataset contains duplicate values
def check_duplicates(df):
    duplicate_count = df.duplicated().sum()
    assert duplicate_count == 0, f"There are {duplicate_count} duplicate values in the DataFrame"

# Test if all columns have positive values
def test_positive_values(df):
    assert (df >= 0).all().all(), "All columns should have positive values"

#--- before data = split_data(df)

# Test if the split_data function returns a tuple with two elements, train_data and valid_data
def test_split_data(df):
    result = split_data(df)
    assert isinstance(result, tuple), "split_data should return a tuple"
    assert len(result) == 2, "split_data should return a tuple with two elements"
    train_data, valid_data = result
    
# Test if train_data contains the correct number of records, based on the percentage split in the function argument
def test_train_data_length(df):
    train_data, valid_data = split_data(df)
    assert len(train_data[0]) == int(len(df) * 0.7), "train_data should contain 70% of the data"

# Test if valid_data contains the correct number of records, based on the percentage split in the function argument.
def test_valid_data_length(df):
    train_data, valid_data = split_data(df)
    assert abs(len(valid_data[0]) - int(len(df) * 0.3)) <= 1, "valid_data should contain 30% of the data"
    
# Test if valid_data contains the correct number of records, based on the percentage split in the function argument.
def test_correct_data_types(df):
    train_data, valid_data = split_data(df)
    expected_dtypes = {'car1_age': 'float', 'car1_price_mean': 'float', 'age_mean': 'float', 'car1_ccm_mean': 'float', 'car1_claim_mean': 'float', 'car1_sumcl_mean': 'float'}
    for col, dtype in expected_dtypes.items():
        assert train_data[0][col].dtype == dtype, f"train_data column {col} should have data type {dtype}"
        assert valid_data[0][col].dtype == dtype, f"valid_data column {col} should have data type {dtype}"


# In[5]:


#--- before model = train_model(data)
# Test if each RMSE value in the dictionary is a number.
def test_rmse_type(model):
    rmse = model[1]
    assert isinstance(rmse, (int, float)), f"RMSE value should be a number"
    
# Test if the train_model function returns a tuple with correct number of elements.
def test_model_type(model):
    model_name = model[0]
    assert isinstance(model_name, RandomForestRegressor) or isinstance(model_name, GradientBoostingRegressor) or isinstance(model_name, LinearRegression) or isinstance(model_name, XGBRegressor) or isinstance(model_name, DecisionTreeRegressor) or isinstance(model_name, MLPRegressor) or isinstance(model_name, AdaBoostRegressor) or isinstance(model_name, CatBoostRegressor) or isinstance(model_name, PolynomialFeatures) or isinstance(model_name, StackingRegressor), "train_model should return either RandomForestRegressor or GradientBoostingRegressor or LinearRegression or KNeighborsRegressor or XGBRegressor or DecisionTreeRegressor or MLPRegressor or AdaBoostRegressor or CatBoostRegressor or PolynomialFeatures or StackingRegressor"

'''
# Test if the models run without errors    
def test_train_model(data):
    try:
        train_model(data)
    except Exception as e:
        assert False, f"Error occurred while training the models: {str(e)}"
    assert True
'''    

# Test if the rmse aren't exceeded
def test_max_rmse(model):
    rmse = model[1]
    max_rmse = 50
    assert max_rmse >= rmse, f"RMSE value should be lesser than or equal to {max_rmse}"


# In[2]:


# Train the model, return the model
def train_model(data):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]
    #The variables train_data.data and trai n_data.label represent the x_train and x_test variables, respectively

    x_train = data[0][0]
    x_test = data[1][0]
    y_train = data[0][1]
    y_test = data[1][1]

    #x_train = train_data.data
    #x_test = valid_data.data
    #y_train = train_data.label
    #y_test = valid_data.label
    
    #Cross Validation Iterator
    n_splits = 10 # Define the number of folds for cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # Create a k-fold cross-validation object 
    
    rf_model, mean_rmse_scores_rf = rf_regressor(x_train, y_train, kf) #Random Forest Regressor
    gb_model, mean_rmse_scores_gb = gb_regressor(x_train, y_train, kf) #Gradient Boosting Regressor
    dnn_model, mean_rmse_scores_dnn = dnn_regressor(x_train, y_train, kf) #DNN Regressor
    lr_model, mean_rmse_scores_lr = lr_regressor(x_train, y_train, kf) #Linear Regressor
    xgb_model, mean_rmse_scores_xgb = xgb_regressor(x_train, y_train, kf) #XGBoost Regressor
    dt_model, mean_rmse_scores_dt = dt_regressor(x_train, y_train, kf) #Decision Tree Regressor
    mlp_model, mean_rmse_scores_mlp = mlp_regressor(x_train, y_train, kf) #ANN Regressor
    ada_model, mean_rmse_scores_ada = adaboost_regressor(x_train, y_train, kf) #Adaboost Regressor
    cat_model, mean_rmse_scores_cat = catboost_regressor(x_train, y_train, kf) #CatBoost Regressor
    poly_model, mean_rmse_scores_poly = poly_regressor(x_train, y_train, kf) #Polynomial Regressor
    stack_model, mean_rmse_scores_stack = stack_regressor(x_train, y_train, kf) #Stacking Regressor
        
    # Calculate the mean score for each metric across all folds
    mean_rmse_rf = round(pd.Series(mean_rmse_scores_rf).mean(), 4)
    mean_rmse_gb = round(pd.Series(mean_rmse_scores_gb).mean(), 4)
    mean_rmse_dnn = round(pd.Series(mean_rmse_scores_dnn).mean(), 4)
    mean_rmse_lr = round(pd.Series(mean_rmse_scores_lr).mean(), 4)
    mean_rmse_xgb = round(pd.Series(mean_rmse_scores_xgb).mean(), 4)
    mean_rmse_dt = round(pd.Series(mean_rmse_scores_dt).mean(), 4)
    mean_rmse_mlp = round(pd.Series(mean_rmse_scores_mlp).mean(), 4)
    mean_rmse_ada = round(pd.Series(mean_rmse_scores_ada).mean(), 4)
    mean_rmse_cat = round(pd.Series(mean_rmse_scores_cat).mean(), 4)
    mean_rmse_poly = round(pd.Series(mean_rmse_scores_poly).mean(), 4)
    mean_rmse_stack = round(pd.Series(mean_rmse_scores_stack).mean(), 4)
    
    # Create a dictionary to store the models and their RMSE values
    models = {'rf': (rf_model, mean_rmse_rf),
              'gb': (gb_model, mean_rmse_gb),
              'dnn': (dnn_model, mean_rmse_dnn),
              'lr': (lr_model, mean_rmse_lr),
              'xgb': (xgb_model, mean_rmse_xgb),
              'dt': (dt_model, mean_rmse_dt), 
              'mlp': (mlp_model, mean_rmse_mlp),
              'ada': (ada_model, mean_rmse_ada),
              'cat': (cat_model, mean_rmse_cat),
              'poly': (poly_model, mean_rmse_poly),
              'stack': (stack_model, mean_rmse_stack)} 

    # Find the model with the lowest RMSE using the built-in min() function
    best_model, best_rmse = min(models.values(), key=lambda x: x[1])

    return best_model, best_rmse


# In[ ]:


def rf_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        rf_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        rf_pred_fold = rf_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        rf_rmse_fold = mean_squared_error(y_test_fold, rf_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(rf_rmse_fold)
        
    return rf_model, mean_rmse_scores


# In[ ]:


def gb_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    gb_model = GradientBoostingRegressor()
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        gb_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        gb_pred_fold = gb_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        gb_rmse_fold = mean_squared_error(y_test_fold, gb_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(gb_rmse_fold)
        
    return gb_model, mean_rmse_scores


# In[ ]:

def dnn_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    # Define the model architecture
    dnn_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(x_train.columns)]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    # Compile the model with mean squared error as the loss function and Adam optimizer
    dnn_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    
    # Define early stopping criteria
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        dnn_model.fit(X_train_fold, y_train_fold, epochs=60, verbose=0)
        
        # Make predictions on this fold's testing data
        dnn_pred_fold = dnn_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        dnn_rmse_fold = mean_squared_error(y_test_fold, dnn_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(dnn_rmse_fold)
        
    return dnn_model, mean_rmse_scores



# In[ ]:


def lr_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    lr_model = LinearRegression()
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        lr_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        lr_pred_fold = lr_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        lr_rmse_fold = mean_squared_error(y_test_fold, lr_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(lr_rmse_fold)
        
    return lr_model, mean_rmse_scores


# In[ ]:


def xgb_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    xgb_model = XGBRegressor(random_state=42)
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        xgb_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        xgb_pred_fold = xgb_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        xgb_rmse_fold = mean_squared_error(y_test_fold, xgb_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(xgb_rmse_fold)
        
    return xgb_model, mean_rmse_scores


# In[ ]:


def dt_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    dt_model = DecisionTreeRegressor(random_state=42)
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        dt_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        dt_pred_fold = dt_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        dt_rmse_fold = mean_squared_error(y_test_fold, dt_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(dt_rmse_fold)
        
    return dt_model, mean_rmse_scores


# In[ ]:


def mlp_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    mlp_model = MLPRegressor()
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        mlp_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        mlp_pred_fold = mlp_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        mlp_rmse_fold = mean_squared_error(y_test_fold, mlp_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(mlp_rmse_fold)
        
    return mlp_model, mean_rmse_scores


# In[ ]:


def adaboost_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    adaboost_model = AdaBoostRegressor(n_estimators=40, random_state=42)
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        adaboost_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        adaboost_pred_fold = adaboost_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        adaboost_rmse_fold = mean_squared_error(y_test_fold, adaboost_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(adaboost_rmse_fold)
        
    return adaboost_model, mean_rmse_scores


# In[ ]:


def catboost_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    catboost_model = CatBoostRegressor(random_state=42, verbose=False)
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        catboost_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        catboost_pred_fold = catboost_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        catboost_rmse_fold = mean_squared_error(y_test_fold, catboost_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(catboost_rmse_fold)
        
    return catboost_model, mean_rmse_scores


# In[ ]:


def poly_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    poly_model = PolynomialFeatures(degree=2)
    poly_model_train = LinearRegression()
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        #Train the Polynomial Regression model using the training data
        x_poly_train = poly_model.fit_transform(X_train_fold)
        
        # Fit the model on this fold's training data
        poly_model_train.fit(x_poly_train, y_train_fold)
        
        # Make predictions on this fold's testing data
        x_poly_test = poly_model.transform(X_test_fold)
        poly_pred_fold = poly_model_train.predict(x_poly_test)
        
        # Calculate the metrics for this fold's predictions
        poly_rmse_fold = mean_squared_error(y_test_fold, poly_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(poly_rmse_fold)
        
    return poly_model, mean_rmse_scores


# In[ ]:


def stack_regressor(x_train, y_train, kf):
    # Create lists to store mean scores
    mean_rmse_scores = []
    
    # Train the Stacking Regressor using the Random Forest and Gradient Boosting models
    # Define the base level regressors
    xgb = XGBRegressor()
    vr = VotingRegressor(estimators=[('xgb', xgb), ('lr', LinearRegression())])
    
    # Define the second level regressor
    catboost_model = CatBoostRegressor()
    
    # Define the stacking regressor
    stack_model = StackingRegressor(
        estimators=[('level0', vr), ('level1', catboost_model)],
        final_estimator=LinearRegression())
    
    # Loop through each split of KFold
    for train_index, test_index in kf.split(x_train, y_train):
    
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model on this fold's training data
        stack_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions on this fold's testing data
        stack_pred_fold = stack_model.predict(X_test_fold)
        
        # Calculate the metrics for this fold's predictions
        stack_rmse_fold = mean_squared_error(y_test_fold, stack_pred_fold, squared=False)
        
        # Append the mean score for this fold to the appropriate list
        mean_rmse_scores.append(stack_rmse_fold)
        
    return stack_model, mean_rmse_scores


# In[6]:


# Evaluate the metrics for the model
def get_model_metrics(model):
    """Construct a dictionary of metrics for the model"""

    # Compute the RMSE on the validation set
    rmse = model[1]

    model_metrics = {
        "rmse": rmse
    }

    print(model_metrics)

    return model_metrics

