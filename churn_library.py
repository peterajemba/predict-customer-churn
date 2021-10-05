# library doc string
'''
This library contains code that can be used to Predict Customer Churn.

Author: Peter Ajemba
Date: October 2021
'''

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    # Import data
    data_frame = pd.read_csv(pth)

    # Add response variable
    data_frame[
        'Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    return data_frame


def perform_eda(data_frame, pth):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe
            pth: directory to save the figures

    output:
            None
    '''
    # Generate histograms for Churn
    hist_churn = data_frame['Churn'].hist()
    fig_churn = hist_churn.get_figure()
    fig_churn.savefig([pth + "/Churn_Histogram.png"])

    # Generate histograms for Customer Age
    hist_age = data_frame['Customer_Age'].hist()
    fig_age = hist_age.get_figure()
    fig_age.savefig([pth + "/Customer_Age_Histogram.png"])

    # Generate histogram for Marital Status
    hist_marital_status = data_frame.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    fig_marital_status = hist_marital_status.get_figure()
    fig_marital_status.savefig([pth + "/Marital_Status_Histogram.png"])

    # Generate heatmap of all features
    hist_heatmap = sns.heatmap(
        data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig_heatmap = hist_heatmap.get_figure()
    fig_heatmap.savefig([pth + "/Heatmap.png"])

    # Generate histogram for Total Trans Count
    # plt.figure(figsize=(20,10))
    hist_total_trans_ct = sns.distplot(data_frame['Total_Trans_Ct'])
    fig_total_trans_ct = hist_total_trans_ct.get_figure()
    fig_total_trans_ct.savefig([pth + "/Total_Trans_Ct_Plot.png"])


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                      used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        output_lst = []
        mean_groups = data_frame.groupby(category).mean()[response]
        for value in data_frame[category]:
            output_lst.append(mean_groups.loc[value])
        data_frame[category + '_' + response] = output_lst

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    Performs feature engineering and creates the training and testing input data

    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create the target variable
    y = data_frame['Churn']

    # Create a container for the feature vector
    X = pd.DataFrame()
    X[response] = data_frame[response]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Declare the Grid Search Parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Create the Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)

    # Train the Random Forest Classifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Create the Logistic Regression Classifier
    lrc = LogisticRegression()

    # Train the Logistic Regression Classifier
    lrc.fit(X_train, y_train)


def generate_classifier_predictions(pth, X_train, X_test, y_train, y_test):
    '''
    Generates training and testing results from applying model to train and
    test datasets.

    input:
            pth: a path to the classifiers
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data

    Output:
            y_train_preds_rf: Random Forest y train predict
            y_test_preds_rf: Random Forest y test predict
            y_train_preds_lr: Linear Regression y train predict
            y_test_preds_lr: Linear Regression y test predict
    '''
    # Import Random Forest Classifier
    cv_rfc = None

    # Import Logistic Regression Classifier
    lrc = None

    # Predict using Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Predict using Logistic Regression Classifier
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


if __name__ == "__main__":
    # Import data
    df = import_data("./data/bank_data.csv")
    print("Dataframe shape is {}".format(df.shape))

    # Perform EDA
    perform_eda(df, "./images/eda")
    print("Dataframe shape is {}".format(df.shape))

    # Define response string
    selection = 'Churn'

    # Define categories to encode
    categories = ['Gender', 'Education_Level', 'Marital_Status',
                  'Income_Category', 'Card_Category']

    # Encode selected categories
    df = encoder_helper(df, categories, selection)
    print("Dataframe shape is {}".format(df.shape))

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn']
