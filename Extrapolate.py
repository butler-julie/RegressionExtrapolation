##################################################
# Extrapolate
# Part of the Library: Sequential Regression Extrapolation
# Julie Butler Hartley
# Version 1.0.0
# Date Created: February 28, 2021
# Last Modified: February 28, 2021
# 
# A collection of functions that perform sequential regression extrapolation
# on a given data set.
##################################################

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
import numpy as np
# LOCAL IMPORTS
from Regression import *
from Support import *
import Analysis

##############################
# SEQUENTIAL EXTRAPOLATE
##############################
def sequential_extrapolate(R, y_train, num_points, seq=2,\
    isAutoRegressive = False, isErrorAnalysis = False, y_true = []):
    """
        Inputs:
            R (a regression model): An instance of one of the regression
                classes in Regression.py
            y_train (a list): the training data used to fit R
            num_points (an int): the total number of points to be in the final
                extrapolated list
            seq (an int): the length of sequence to be used.  This needs to be
                the same length of sequence that was used when fitting the
                model.  Default value is 2.
            isErrorAnalysis (a boolean): True case means error analysis 
                information is printed to the console comparing the
                predicted values to the known values.
            y_true (a list): the known data set for comparison
        Returns:
            y_test (a list): the predicted, extrapolated values
        Performs sequential regression extrapolation on a given data set with
        one row.
    """
    # Make sure inputs are of the proper type
    assert isinstance(num_points, int)
    assert isinstance(seq, int)
    assert isinstance(isErrorAnalysis, bool)
    # If the given regression model has not yet been trained then it can not
    # be used for extrapolation.  End the program
    if not R.isFit:
        print("Model is not trained so can not be used for extrapolation")
        print("Use the fit method to train a model prior to extrapolation")
        print("Program will terminiate.")
        import sys
        sys.exit()
    # If the model has been trained the perform the extrapolation
    y_test = y_train.copy()
    # Extrapolate until enough data points have been predicted
    while len(y_test) < num_points:
        next_test = y_test[-seq:]
        point = R.predict([next_test])
        y_test.append(point[0])
        if isAutoRegressive:
            X_train, y_train = format_sequential_data(y_test, seq)
            R.fit(X_train, y_train)
    # Perform error analysis if needed
    if isErrorAnalysis:
        EA = Analysis.ErrorAnalysis()
        print("The MSE score between the predicted and true data is", EA.mse(np.asarray(y_test), np.asarray(y_true)))
        try:
            print("The R2 score between the predicted and true data is", EA.r2(np.asarray(y_test), np.asarray(y_true)))
        except:
                print("Cannot calculate R2 Score")
    # Return the predicted data set
    return y_test

##############################
# SEQUENTIAL COLUMN EXTRAPOLATE
##############################
def sequential_column_extrapolate (R, formattedData, num_new_cols, seq=2, isAutoRegressive=False):
    """
        Inputs:
            R (a class instance): An instance of one of the regression classes
                for sequential data extrapolation (i.e. one of the classes
                from Regression.py)
            formattedData (a 2D numpy array): the matrix to be used as the training
                data (its columns are what is to be extrapolated).
            num_new_cols (an int): the number of columns needed in the final,
                extrapolated matrix.
        Returns:
            extrapolated_data (a 2D numpy array): the matrix with the correct number
                of columns, generated through sequential regression extrapolation.
        Performs sequential regression analysis on each row of a given matrix to 
        create a matrix with the desired number of columns.
    """
    # Create a 2D array of zeros to hold the new, extrapolated matrix   
    extrapolated_data = np.zeros((len(formattedData), num_new_cols))
    weights = []
    rows = np.arange(0, len(formattedData))
    # Iterate through each row of the training matrix
    for i in rows:
        # Extract the current row and format it to be used as training data
        row = formattedData[i].copy()
        X_train, y_train = format_sequential_data(row, seq)
        # Using the current row as training data, extrapolate until the correct
        # length is reached (given by num_new_cols)
        R.fit(X_train, y_train)
        weights.append(R.get_weights())
        new_row = sequential_extrapolate(R, row, num_new_cols, seq=seq, isAutoRegressive=isAutoRegressive)
        # Place the extrapolated row in the correct spot in the new matrix
        extrapolated_data[i] = new_row
    # Return the extrapolated matrix    
    return extrapolated_data, weights
