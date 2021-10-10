# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is designed to test skill developed in the **Clean Code Principles** section of the ML DevOps Engineer Nanodegree Udacity course. The goal of the project is to convert a messy notebook that implements a model designed to identify credit card customers that are most likely to churn into a Python package for a machine learning that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package also has the flexibility of being run interactively or from the command-line interface (CLI).

The output consists of the following files:

1. `churn_library.py` a Python library containing functions needed to complete the model development and testing process.

2. `churn_script_logging_and_tests.py` a Python file containing tests and logging to test the functions of the library and log any errors that occur.

3. `churn_notebook.ipynb` containing a refactored version of the code in Notebook form.

The code in both Python files was formatted using [autopep8](https://pypi.org/project/autopep8/), and both scripts provided [pylint](https://pypi.org/project/pylint/) scores of close to **10.0/10.0**.

## Running Files
The development environment for this project used Python **3.x**. In addition to a base Python installation, install additional libraries by using typing the follwing commands in a terminal:
```
pip install scikit-learn==0.22 shap pylint au
```

To test the code, run the file `churn_script_logging_and_tests.py`. The folder `/logs` should contain a file `churn_library.log` with a log of the code run output.
```
ipython churn_script_logging_and_tests.py
```

To run the library directly, the following code will run all the function in the refactored code from the refactored notebook.
```
ipython churn_library.py
```

To check the pylint scores, run the following commands:
```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

Note that the files were formated using the following commands:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

### Successful Completion
1. If successful, completion of the code run would be indicated in the `churn_library.log` file with the following enteries
> Testing import_data: SUCCESS.
> Testing perform_eda: SUCCESS.
> Testing encoder_helper: SUCCESS.
> Testing perform_feature_engineering: SUCCESS.
> Testing train_random_forest_model: SUCCESS.
> Testing train_logistic_regression_model: SUCCESS.
> Testing characterize_logistic_regression_model: SUCCESS.
> Testing characterize_random_forest_model: SUCCESS.
> Testing shapely_plot: SUCCESS.
> Testing feature_importance_plot: SUCCESS.
> Testing plot_two_model_roc_curves: SUCCESS.

2. Review the folder `/models` to view the two models created
 - `Random Forest Classfier` model
 - `Logistic Regression Classifier` model
 
3. Review the folder `/images/eda` to view the following diagrams
 - `Churn` histogram
 - `Customer Age` histogram
 - `Marital Status` histogram
 - `Total Trans Count` histogram
 - `Heatmap`
