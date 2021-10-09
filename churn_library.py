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
    Return dataframe for the csv found at path

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
    Perform eda on data_frame and display/save figures to images folder

    input:
            data_frame: pandas dataframe
            path: a path the directory to save the figures
            save: True to save, False to show

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


def encoder_helper(data_frame, categories, response):
    '''
    Turn each categorical column into a new column with the proportion of
    churn for each category

    input:
            data_frame: pandas dataframe
            categories: list of categorical features for one-hot encoding
            response: classifer target feature

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in categories:
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
            selected_features: selected featuers for classifier training
            response: classifier target feature

    output:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
    '''
    # Create the target variable
    y_target = data_frame[response]

    # Create a container for the feature vector
    x_features = pd.DataFrame()
    x_features[selected_features] = data_frame[selected_features]

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.3, random_state=42)

    # Package train/test data into a dictionary
    model_data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test}

    return model_data


def train_classifiers(model_data, search_param, path):
    '''
    Train and store Random Forest and Logistic Regression classifiers

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
            search_param: grid search parameters
            path: a path to the classifier

    output:
            None
    '''
    # Train the Random Forest Classifier
    train_random_forest_model(model_data, search_param, path)

    # Train the Logistic Regression Classifier
    train_logistic_regression_model(model_data, path)


def train_random_forest_model(model_data, search_param, path):
    '''
    Train and store Random Forest Classifer model to a given path

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
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
    cv_rand_forest.fit(model_data['x_train'], model_data['y_train'])

    # Save the best Random Forest Classifier Model
    joblib.dump(
        cv_rand_forest.best_estimator_,
        path + "/random_forest_model.pkl")


def train_logistic_regression_model(model_data, path):
    '''
    Train and store Logistic Regression Classifer model to a given path

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
            path: a path to the classifier

    output:
            None
    '''
    # Train the Logistic Regression Classifier
    log_reg = LogisticRegression()
    log_reg.fit(model_data['x_train'], model_data['y_train'])

    # Save the best Logistic Regression Model
    joblib.dump(log_reg, path + "/logistic_regression_model.pkl")


def characterize_random_forest_model(
        model_data,
        model_path,
        image_path,
        save=True):
    '''
    Characterize performance of Random Forest model

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
            model_path: a path to the classifier
            image_path: a path to store characterization results
            save: True to save, False to show

    output:
            None
    '''
    # Load Random Forest Classifier
    rand_forest_model = joblib.load(model_path + "/random_forest_model.pkl")

    # Predict using Random Forest Classifier
    y_train_preds_rf = rand_forest_model.predict(model_data['x_train'])
    y_test_preds_rf = rand_forest_model.predict(model_data['x_test'])

    # Create image of model characterization report
    plt.figure(figsize=(5, 5))
    #plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                model_data['y_test'], y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                model_data['y_train'], y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    if save is True:
        plt.savefig(image_path + "/Random_Forest_Metrics.png")
    else:
        plt.show()


def characterize_logistic_regression_model(
        model_data, model_path, image_path, save=True):
    '''
    Characterize performance of Logistic Regression model

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
            model_path: a path to the classifier
            image_path: a path to store characterization results
            save: True to save, False to show

    output:
            None
    '''
    # Load the Logistic Regression model
    log_reg_model = joblib.load(model_path + "/logistic_regression_model.pkl")

    # Predict using Logistic Regression model
    y_train_preds_lr = log_reg_model.predict(model_data['x_train'])
    y_test_preds_lr = log_reg_model.predict(model_data['x_test'])

    # Create image of model characterization report
    plt.figure(figsize=(5, 5))
    #plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                model_data['y_train'], y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                model_data['y_test'], y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    if save is True:
        plt.savefig(image_path + "/Logistic_Regression_Metrics.png")
    else:
        plt.show()


def plot_roc_curves(
        model_data,
        model_one_path,
        model_two_path,
        output_path,
        save=True):
    '''
    Produces ROC curves for both models, plots them individually and together

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
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
    model_one_disp = plot_roc_curve(
        model_one,
        model_data['x_test'],
        model_data['y_test'])
    if save is True:
        plt.savefig(output_path + "/Model_One_ROC.png")
    else:
        plt.show()

    # Plot model two individual ROC curve
    plt.figure(figsize=(15, 8))
    model_two_disp = plot_roc_curve(
        model_two,
        model_data['x_test'],
        model_data['y_test'])
    if save is True:
        plt.savefig(output_path + "/Model_Two_ROC.png")
    else:
        plt.show()

    # Plot model two individual ROC curve
    plt.figure(figsize=(15, 8))
    model_one_disp = plot_roc_curve(
        model_one,
        model_data['x_test'],
        model_data['y_test'])
    model_two_disp = plot_roc_curve(
        model_two,
        model_data['x_test'],
        model_data['y_test'],
        ax=model_one_disp.ax_)
    model_two_disp.figure_.suptitle("ROC Curve Comparison")
    if save is True:
        plt.savefig(output_path + "/Combined_ROC.png")
    else:
        plt.show()


def shapely_plot(model_data, model_path, output_path, save=True):
    '''
    Creates and stores the feature importances in path

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
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
    shapely_values = explainer.shap_values(model_data['x_test'])

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Shapely Summary Plot")
    shap.summary_plot(shapely_values, model_data['x_test'], plot_type="bar")
    if save is True:
        plt.savefig(output_path + "/Shapely_Summary_Plot.png")
    else:
        plt.show()


def feature_importance_plot(model_data, model_path, output_path, save=True):
    '''
    creates and stores the feature importances in path

    input:
            model_data: contains X training data (x_train), X testing data
                        (x_test), y training data (y_train) and y testing data
                        (y_test)
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
    names = [model_data['x_test'].columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(model_data['x_test'].shape[1]), importances[indices])
    plt.xticks(range(model_data['x_test'].shape[1]), names, rotation=90)
    if save is True:
        plt.savefig(output_path + "/Feature_Importance_Plot.png")
    else:
        plt.show()


if __name__ == "__main__":
    # import os
    import os

    # Import data
    df = import_data("./data/bank_data.csv")
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    print("Testing import_data: SUCCESS")

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
    print("Testing perform_eda: SUCCESS")

    # Identify response variable
    RESPONSE = 'Churn'

    # Define one-hot encoding categories
    CATEGORIES = ['Gender', 'Education_Level', 'Marital_Status',
                  'Income_Category', 'Card_Category']

    # Encode selected categories
    df = encoder_helper(df, CATEGORIES, RESPONSE)
    assert df.shape[0] == 10127
    assert df.shape[1] == 28
    print("Testing encoder_helper: SUCCESS")

    # Identify features to use for classifier training
    SELECTED_FEATURES = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn']

    # Generate train/test data
    MODEL_DATA = perform_feature_engineering(
        df, SELECTED_FEATURES, RESPONSE)
    assert MODEL_DATA['x_train'].shape[0] > 0
    assert MODEL_DATA['x_train'].shape[1] > 0
    assert MODEL_DATA['x_test'].shape[0] > 0
    assert MODEL_DATA['x_test'].shape[1] > 0
    assert MODEL_DATA['y_train'].shape[0] > 0
    assert MODEL_DATA['y_test'].shape[0] > 0
    print("Testing perform_feature_engineering: SUCCESS")

    # Define Random Forest grid search parameters
    SEARCH_PARAM = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Train Random Forest classifier
    train_random_forest_model(MODEL_DATA, SEARCH_PARAM, "./models")
    assert os.path.exists("./models/random_forest_model.pkl") is True
    print("Testing train_random_forest_model: SUCCESS")

    # Train Logistic Regression classifier
    train_logistic_regression_model(MODEL_DATA, "./models")
    assert os.path.exists("./models/logistic_regression_model.pkl") is True
    print("Testing train_logistic_regression_model: SUCCESS")

    # Characterize Random Forest model
    characterize_random_forest_model(
        MODEL_DATA,
        "./models",
        "./images/results",
        save=True)
    assert os.path.exists("./images/results/Random_Forest_Metrics.png") is True
    print("Testing characterize_random_forest_model: SUCCESS")

    # Characterize Logistic Regression model
    characterize_logistic_regression_model(
        MODEL_DATA, "./models", "./images/results", save=True)
    assert os.path.exists(
        "./images/results/Logistic_Regression_Metrics.png") is True
    print("Testing characterize_logistic_regression_model: SUCCESS")

    # Plot ROC curves for Random Forest and Logistic Regression models
    plot_roc_curves(
        MODEL_DATA,
        "./models/random_forest_model.pkl",
        "./models/logistic_regression_model.pkl",
        "./images/results",
        save=True)
    assert os.path.exists("./images/results/Model_One_ROC.png") is True
    assert os.path.exists("./images/results/Model_Two_ROC.png") is True
    assert os.path.exists("./images/results/Combined_ROC.png") is True
    print("Testing plot_two_model_roc_curves: SUCCESS")

    # Plot Shapely summary plot for Random Forest model
    shapely_plot(
        MODEL_DATA,
        "./models/random_forest_model.pkl",
        "./images/results/",
        save=True)
    assert os.path.exists("./images/results/Shapely_Summary_Plot.png") is True
    print("Testing shapely_plot: SUCCESS")

    # Plot Feature Importance plot for Random Forest model
    feature_importance_plot(
        MODEL_DATA,
        "./models/random_forest_model.pkl",
        "./images/results/",
        save=True)
    assert os.path.exists(
        "./images/results/Feature_Importance_Plot.png") is True
    print("Testing feature_importance_plot: SUCCESS")
