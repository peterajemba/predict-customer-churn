# library doc string
'''
This library contains code that can be used to Predict Customer Churn.
'''

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    # Read the dataframe
    data_frame = pd.read_csv(pth)
    
    # Add column to represent Churn
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    # Create and save histogram for Churn
    #plt.figure(figsize=(20,10))
    hist_churn = data_frame['Churn'].hist()
    fig_churn = hist_churn.get_figure()
    fig_churn.savefig("./images/eda/Churn_Histogram.png")
    
    # Create and save histogram for Customer Age
    #plt.figure(figsize=(20,10))
    hist_customer_age = data_frame['Customer_Age'].hist()
    fig_customer_age = hist_customer_age.get_figure()
    fig_customer_age.savefig("./images/eda/Customer_Age_Histogram.png")
    
    # Create and save histogram for Marital Status
    #plt.figure(figsize=(20,10)) 
    hist_marital_status = data_frame.Marital_Status.value_counts('normalize').plot(kind='bar');
    fig_marital_status = hist_marital_status.get_figure()
    fig_marital_status.savefig("./images/eda/Marital_Status_Histogram.png")
    
    # Create and save histogram for Total Trans Count
    #plt.figure(figsize=(20,10))
    hist_total_trans_ct = sns.displot(data_frame['Total_Trans_Ct'])
    fig_total_trans_ct = hist_total_trans_ct.get_figure()
    fig_total_trans_ct.savefig("./images/eda/Total_Trans_Ct_Plot.png")
    
    # Create and save heatmap of all features
    #plt.figure(figsize=(20,10))
    hist_heatmap = sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig_heatmap = hist_heatmap.get_figure()
    fig_heatmap.savefig(r"./images/eda/Heatmap.png")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
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
    pass

if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)