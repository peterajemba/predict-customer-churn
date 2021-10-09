'''
This test file contains code that tests the library to be used to Predict Customer Churn.

Author: Peter Ajemba
Date: October 2021
'''

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

BANK_DATA_PATH = "./data/bank_data.csv"
EDA_IMAGES_PATH = "./images/eda"
MODEL_PATH = "./models"
RESULT_IMAGES_PATH = "./images/results"

# Identify response variable
RESPONSE = 'Churn'

# Define response selection and categories for one-hot encoding
CATEGORIES = ['Gender', 'Education_Level', 'Marital_Status',
              'Income_Category', 'Card_Category']

# Define features selected from feature engineering
selected_features = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn']

# Define Random Forest grid search parameters
search_param = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}


def test_import(import_data):
    '''
    test data import
    '''
    logging.info("I'm in")
    try:
        data_frame = import_data(BANK_DATA_PATH)
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found.")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        logging.info("Testing import_data: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't have rows and columns.")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(cl.import_data(BANK_DATA_PATH), EDA_IMAGES_PATH, save=True)
    except FileNotFoundError as err:
        logging.error("Testing perform eda: EDA plots not printed.")
        raise err

    try:
        assert os.path.exists(EDA_IMAGES_PATH + "/Heatmap.png") is True
        assert os.path.exists(
            EDA_IMAGES_PATH + "/Churn_Histogram.png") is True
        assert os.path.exists(
            EDA_IMAGES_PATH + "/Customer_Age_Histogram.png") is True
        assert os.path.exists(
            EDA_IMAGES_PATH + "/Marital_Status_Histogram.png") is True
        assert os.path.exists(
            EDA_IMAGES_PATH + "/Total_Trans_Ct_Plot.png") is True
        logging.info("Testing perform_eda: SUCCESS.")
    except AssertionError as err:
        logging.error("Testing perform eda: One or more plots not saved.")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        data_frame = cl.import_data(BANK_DATA_PATH)
        assert data_frame.shape[1] == 23
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: DataFrame improperly imported.")
        raise err

    try:
        data_frame = encoder_helper(data_frame, CATEGORIES, RESPONSE)
        assert data_frame.shape[1] == 28
        logging.info("Testing encoder_helper: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: New encodings incorrectly added.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        data_frame = cl.import_data(BANK_DATA_PATH)
        data_frame = cl.encoder_helper(data_frame, CATEGORIES, RESPONSE)
        model_data = perform_feature_engineering(
            data_frame, selected_features, RESPONSE)
        assert model_data['x_train'].shape[0] > 0
        assert model_data['x_train'].shape[1] > 0
        assert model_data['x_test'].shape[0] > 0
        assert model_data['x_test'].shape[1] > 0
        assert model_data['y_train'].shape[0] > 0
        assert model_data['y_test'].shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Incorrect train/test data output.")
        raise err


def test_train_classifiers(train_classifiers):
    '''
    test train_random_forest_model
    '''
    # Create input data and build models
    try:
        data_frame = cl.import_data(BANK_DATA_PATH)
        data_frame = cl.encoder_helper(data_frame, CATEGORIES, RESPONSE)
        model_data = cl.perform_feature_engineering(
            data_frame, selected_features, RESPONSE)
        train_classifiers(model_data, search_param, MODEL_PATH)
        assert os.path.exists(MODEL_PATH + "/random_forest_model.pkl") is True
        logging.info("Testing train_random_forest_model: SUCCESS.")
        assert os.path.exists(MODEL_PATH +
                              "/logistic_regression_model.pkl") is True
        logging.info("Testing train_logistic_regression_model: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing train_classifiers: Models not created.")
        raise err

    # Train, test and characterize models
    try:
        cl.characterize_logistic_regression_model(
            model_data, MODEL_PATH, RESULT_IMAGES_PATH, save=True)
        assert os.path.exists(
            RESULT_IMAGES_PATH +
            "/Logistic_Regression_Metrics.png") is True
        logging.info("Testing characterize_logistic_regression_model: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing characterize_logistic_regression_model: Result image not created.")
        raise err

    try:
        cl.characterize_random_forest_model(
            model_data, MODEL_PATH, RESULT_IMAGES_PATH, save=True)
        assert os.path.exists(
            RESULT_IMAGES_PATH +
            "/Random_Forest_Metrics.png") is True
        logging.info("Testing characterize_random_forest_model: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing characterize_random_forest_model: Result image not created.")
        raise err

    # Plot Shapely summary plot for Random Forest model
    try:
        cl.shapely_plot(
            model_data,
            MODEL_PATH +
            "/random_forest_model.pkl",
            RESULT_IMAGES_PATH,
            save=True)
        assert os.path.exists(
            RESULT_IMAGES_PATH +
            "/Shapely_Summary_Plot.png") is True
        logging.info("Testing shapely_plot: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing shapely_plot: Result image not created.")
        raise err

    # Plot Feature Importance plot for Random Forest model
    try:
        cl.feature_importance_plot(
            model_data,
            MODEL_PATH +
            "/random_forest_model.pkl",
            RESULT_IMAGES_PATH,
            save=True)
        assert os.path.exists(
            RESULT_IMAGES_PATH +
            "/Feature_Importance_Plot.png") is True
        logging.info("Testing feature_importance_plot: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: Result image not created.")
        raise err

    try:
        cl.plot_roc_curves(
            model_data,
            MODEL_PATH +
            "/random_forest_model.pkl",
            MODEL_PATH +
            "/logistic_regression_model.pkl",
            RESULT_IMAGES_PATH,
            save=True)
        assert os.path.exists(
            RESULT_IMAGES_PATH +
            "/Model_One_ROC.png") is True
        assert os.path.exists(
            RESULT_IMAGES_PATH +
            "/Model_Two_ROC.png") is True
        assert os.path.exists(RESULT_IMAGES_PATH + "/Combined_ROC.png") is True
        logging.info("Testing plot_two_model_roc_curves: SUCCESS.")
    except AssertionError as err:
        logging.error("Testing plot_roc_curves: Result images not created.")
        raise err


if __name__ == "__main__":

    # Test import_data
    test_import(cl.import_data)

    # Test perform_eda
    test_eda(cl.perform_eda)

    # Test encoder_helper
    test_encoder_helper(cl.encoder_helper)

    # Test encoder_helper
    test_perform_feature_engineering(cl.perform_feature_engineering)

    # Test test_train_models
    test_train_classifiers(cl.train_classifiers)
