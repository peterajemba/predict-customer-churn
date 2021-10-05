'''
This library contains code that tests the library to be used to Predict Customer Churn.

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


def test_import(import_data):
    '''
    test data import
    '''
    logging.info("I'm in")
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found.")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(cl.import_data("./data/bank_data.csv"), "./images/eda")
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform eda: Plots not printed.")
        raise err

    try:
        assert os.path.exists("./images/eda/Heatmap.png") is True
        assert os.path.exists("./images/eda/Churn_Histogram.png") is True
        assert os.path.exists(
            "./images/eda/Customer_Age_Histogram.png") is True
        assert os.path.exists(
            "./images/eda/Marital_Status_Histogram.png") is True
    except AssertionError as err:
        logging.error("Testing perform eda: Plots not printed.")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    response = 'Churn'
    category_lst = [
        'Gender', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category']
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        assert data_frame.shape[0] == 10127
        assert data_frame.shape[1] == 23
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: DataFrame not properly generated.")
        raise err

    try:
        data_frame = encoder_helper(data_frame, category_lst, response)
        assert data_frame.shape[0] == 10127
        assert data_frame.shape[1] == 28
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: New categories incorrectly added.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":

    # Test import_data
    test_import(cl.import_data)

    # Test perform_eda
    test_eda(cl.perform_eda)

    # Test encoder_helper
    test_encoder_helper(cl.encoder_helper)
