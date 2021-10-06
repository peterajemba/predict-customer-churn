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
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            path: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    # Import data
    data_frame = pd.read_csv(path)

    # Add response variable
    data_frame[
        'Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    return data_frame


def perform_eda(data_frame, path, save=True):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe
            pth: directory to save the figures

    output:
            None
    '''
    # Generate histograms for Churn
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    if save is True:
        plt.savefig(path + "/Churn_Histogram.png")
    else:
        plt.show()

    # Generate histograms for Customer Age
    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    if save is True:
        plt.savefig(path + "/Customer_Age_Histogram.png")
    else:
        plt.show()

    # Generate histogram for Marital Status
    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    if save is True:
        plt.savefig(path + "/Marital_Status_Histogram.png")
    else:
        plt.show()

    # Generate heatmap of all features
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    if save is True:
        plt.savefig(path + "/Heatmap.png")
    else:
        plt.show()

    # Generate histogram for Total Trans Count
    plt.figure(figsize=(20, 10))
    sns.distplot(data_frame['Total_Trans_Ct'])
    if save is True:
        plt.savefig(path + "/Total_Trans_Ct_Plot.png")
    else:
        plt.show()


def encoder_helper(data_frame, category_list, response):
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
    for category in category_list:
        output_lst = []
        mean_groups = data_frame.groupby(category).mean()[response]
        for value in data_frame[category]:
            output_lst.append(mean_groups.loc[value])
        data_frame[category + '_' + response] = output_lst

    return data_frame


def perform_feature_engineering(data_frame, selected_features, response):
    '''
    Perform feature engineering and create the training and testing input tables

    input:
              data_frame: pandas dataframe
              selected_features: list of input model features
              response: target features

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create the target variable
    y_target = data_frame[response]

    # Create a container for the feature vector
    x_features = pd.DataFrame()
    x_features[selected_features] = data_frame[selected_features]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def train_random_forest_model(x_train, y_train, search_param, path):
    '''
    train and store Random Forest Classifer to given path
    input:
              x_train: X training data
              y_train: y training data
              search_param: grid search parameters
              path: a path to the classifier
    output:
              None
    '''
    # Train the Random Forest Classifier
    rand_forest = RandomForestClassifier(random_state=42)
    cv_rand_forest = GridSearchCV(
        estimator=rand_forest,
        param_grid=search_param,
        cv=5)
    cv_rand_forest.fit(x_train, y_train)

    # Save the best Random Forest Classifier Model
    joblib.dump(
        cv_rand_forest.best_estimator_,
        path + "/random_forest_model.pkl")


def train_logistic_regression_model(x_train, y_train, path):
    '''
    train, test and store Logistic Regression Classifer model results
    input:
              x_train: X training data
              y_train: y training data
              path: a path to the classifier
    output:
              None
    '''
    # Train the Logistic Regression Classifier
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    # Save the best Logistic Regression Model
    joblib.dump(log_reg, path + "/logistic_regression_model.pkl")


def characterize_random_forest_model(x_train, x_test, y_train, y_test, path, save=True):
    '''
    characterize performance of Random Forest model
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              path: a path to the classifier
              save: save image (True) or display image (False) 
    output:
              None
    '''
    # Load Random Forest Classifier
    rand_forest_model = joblib.load(path + "/random_forest_model.pkl")

    # Predict using Random Forest Classifier
    y_train_preds_rf = rand_forest_model.predict(x_train)
    y_test_preds_rf = rand_forest_model.predict(x_test)
    
    # Create image of model characterization report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off');
    if save is True:
        plt.savefig(path + "/Random_Forest_Metrics.png")
    else:
        plt.show()    


def characterize_logistic_regression_model(
        x_train, x_test, y_train, y_test, path, save=True):
    '''
    characterize performance of Logistic Regression model
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              path: a path to the classifier
              save: save image (True) or display image (False) 
    output:
              None
    '''
    # Load the Logistic Regression model
    log_reg_model = joblib.load(path + "/logistic_regression_model.pkl")

    # Predict using Logistic Regression model
    y_train_preds_lr = log_reg_model.predict(x_train)
    y_test_preds_lr = log_reg_model.predict(x_test)
    
    # Create image of model characterization report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off');
    if save is True:
        plt.savefig(path + "/Logistic_Regression_Metrics.png")
    else:
        plt.show()    


def plot_two_model_roc_curves(
        x_test,
        y_test,
        model_one_path,
        model_two_path,
        output_path,
        save=True):
    '''
    produces ROC curves for both models, plots them individually and together

    input:
            x_test: X testing data
            y_test: y testing data
            model_one_path: first model path
            model_two_path: second model path
            output_path: output_path
            save: True to save, False to show
    output:
             None
    '''
    # Load models
    model_one = joblib.load(model_one_path)
    model_two = joblib.load(model_two_path)

    # Plot model one individual ROC curve
    plt.figure(figsize=(15, 8))
    model_one_disp = plot_roc_curve(model_one, x_test, y_test)
    if save is True:
        plt.savefig(output_path + "/Model_One_ROC.png")
    else:
        plt.show()

    # Plot model two individual ROC curve
    plt.figure(figsize=(15, 8))
    model_two_disp = plot_roc_curve(model_two, x_test, y_test)
    if save is True:
        plt.savefig(output_path + "/Model_Two_ROC.png")
    else:
        plt.show()

    # Plot model two individual ROC curve
    plt.figure(figsize=(15, 8))
    model_one_disp = plot_roc_curve(model_one, x_test, y_test)
    model_two_disp = plot_roc_curve(
        model_two, x_test, y_test, ax=model_one_disp.ax_)
    model_two_disp.figure_.suptitle("ROC Curve Comparison")
    if save is True:
        plt.savefig(output_path + "/Combined_ROC.png")
    else:
        plt.show()


def shapely_plot(x_data, model_path, output_path, save=True):
    '''
    creates and stores the feature importances in path
    input:
            x_data: pandas dataframe of X values
            model_path: path to model object
            output_pth: path to store the figure
            save: True to save, False to show
    output:
             None
    '''
    # Load model
    model = joblib.load(model_path)
    
    # Calculate feature explainer
    explainer = shap.TreeExplainer(model)
    
    # Generate Shapely values
    shapely_values = explainer.shap_values(x_data)
    
    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Shapely Summary Plot")
    shap.summary_plot(shapely_values, x_data, plot_type="bar")
    if save is True:
        plt.savefig(output_path + "Shapely_Summary_Plot.png")
    else:
        plt.show()
        
    
def feature_importance_plot(x_data, model_path, output_path, save=True):
    '''
    creates and stores the feature importances in path
    input:
            x_data: pandas dataframe of X values
            model_path: path to model object
            output_pth: path to store the figure
            save: True to save, False to show
    output:
             None
    '''
    # Load model
    model = joblib.load(model_path)
    
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    if save is True:
        plt.savefig(output_path + "Feature_Importance_Plot.png")
    else:
        plt.show()


if __name__ == "__main__":
    # import os
    import os

    # Import data
    df = import_data("./data/bank_data.csv")
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    print(f"Original Dataframe shape is {df.shape}")

    # Perform EDA
    perform_eda(df, "./images/eda", save=True)
    assert os.path.exists("./images/eda/Heatmap.png") is True
    assert os.path.exists("./images/eda/Churn_Histogram.png") is True
    assert os.path.exists(
        "./images/eda/Customer_Age_Histogram.png") is True
    assert os.path.exists(
        "./images/eda/Marital_Status_Histogram.png") is True
    assert os.path.exists(
        "./images/eda/Total_Trans_Ct_Plot.png") is True

    # Define response selection and categories
    SELECTION = 'Churn'
    categories = ['Gender', 'Education_Level', 'Marital_Status',
                  'Income_Category', 'Card_Category']

    # Encode selected categories
    df = encoder_helper(df, categories, SELECTION)
    assert df.shape[0] == 10127
    assert df.shape[1] == 28
    print(f"Updated Dataframe shape is {df.shape}")

    # Identify features to use for classifier training
    selected_features = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn']
    
    # Identify response variable
    response = 'Churn'

    # Generate train/test data
    x_train, x_test, y_train, y_test = perform_feature_engineering(
                                           df, selected_features, response)
    assert x_train.shape[0] > 0
    assert x_train.shape[1] > 0
    assert x_test.shape[0] > 0
    assert x_test.shape[1] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

    # Train Random Forest classifier
    search_param = {
        'n_estimators' : [200, 500],
        'max_features' : ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' : ['gini', 'entropy']
    }
    train_random_forest_model(x_train, y_train, search_param, "./models")
    assert os.path.exists("./models/random_forest_model.pkl") is True

    # Train Logistic Regression classifier
    train_logistic_regression_model(x_train, y_train, "./models")
    assert os.path.exists("./models/random_forest_model.pkl") is True
    
    # Characterize Random Forest model
    characterize_random_forest_model(x_train, x_test, y_train, y_test, "./models", save=True)
    assert os.path.exists("./models/Random_Forest_Metrics.png") is True
    
    # Characterize Logistic Regression model
    characterize_logistic_regression_model(x_train, x_test, y_train, y_test, "./models", save=True)
    assert os.path.exists("./models/Logistic_Regression_Metrics.png") is True
    
    # Plot ROC curves for Random Forest and Logistic Regression models
    plot_two_model_roc_curves(x_test, y_test, "./models/random_forest_model.pkl", "./models/logistic_regression_model.pkl", "./images/results", save=False)
    assert os.path.exists("./images/results/Model_One_ROC.png") is True
    assert os.path.exists("./images/results/Model_Two_ROC.png") is True
    assert os.path.exists("./images/results/Combined_ROC.png") is True
    
    # Plot Shapely summary plot for Random Forest model
    shapely_plot(x_test, "./models/random_forest_model.pkl", "./images/results/", save=True)
    assert os.path.exists("./images/results/Shapely_Summary_Plot.png") is True
    
    # Plot Feature Importance plot for Random Forest model
    feature_importance_plot(x_test, "./models/random_forest_model.pkl", "./images/results/", save=True)
    assert os.path.exists("./images/results/Feature_Importance_Plot.png") is True
    