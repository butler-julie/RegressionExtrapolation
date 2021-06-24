##################################################
# Support
# Part of the Library: Sequential Regression Extrapolation
# Julie Butler Hartley
# Version 1.0.0
# Date Created: February 28, 2021
# Last Modified: February 28, 2021
##################################################

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
import csv
import numpy as np

##############################
# FORMAT SEQUENTIAL DATA
##############################
def format_sequential_data (y, seq=2):
    """
        Inputs:
            y (a list or NumPy array): the y values of a data set
            seq (an int): the length of the sequence.  Default value is 2
        Returns:
            inputs (a list): the inputs for a machine learning model using 
                sequential data formatting
            outputs (a list): the outputs for a machine learning model using
                sequential data formatting              
        Formats a given list or array in sequential formatting using the 
        given sequence lenght.  Default sequence length is two.

        Explanation of sequential formatting:
        Typically data points of the form (x,y) are used to train a machine
        learning model.  This teaches the model the relationship between the
        x data and the y data in the training range.  This model works well 
        for interpolation, but not so well for extrapolation.  A better data
        model for extrapolation would be one that learns the patterns in the y
        data to better guess what y value should come next.  Therefore, this 
        method formats the data in a sequential pattern so that the points are
        of the form ((y1, y2, ..., yn), yn+1) where n is the lenght of the 
        sequence (seq).
    """
    # Make sure seq is an int
    assert isinstance(seq, int)
    # Set up the input and output lists
    inputs = []
    outputs = []
    # Cycle through the whole y list/array and separate the points into 
    # sequential format
    for i in range(0, len(y)-seq):
        inputs.append(y[i:i+seq])
        outputs.append(y[i+seq])
    # Return the input and output lists.  NOTE: the data type of the return 
    # values is LIST
    return inputs, outputs  


##############################
# FORMAT DATA
##############################
def formatData (filename, delimiter):
    """
        Inputs:
            filename (a string): a file name representing the file where
                the data is stored.  The import code expects the file to 
                contain only the matrix elements with columns separated by
                a set delimiter and rows separated by a new line.
            delimiter (a string): the delimiter between the columns in the 
                text file    
        Returns:
            formattedData (a 2D numpy array): a 2D array storing the matrix. 
                The columns of the matrix can be accessed via 
                formattedData[:,i], and the rows of the matrix can be accessed
                via formattedData[i].
        Imports data from a file and formats it to be used in the matrix 
        extrapolation codes.  Note, this does not format the data in an 
        unusual way.  Many matrices in Python are formatted using this format.
    """
    # Lists to hold the data
    formattedData = []
    val = []
    # Get every row from the file and store them in order in val
    with open(filename, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter=delimiter)
        for row in reader:
            val.append(row)   
    # Convert all if the elements to floats and store in formattedData 
    for row in range(len(val)):
        formattedData.append([float(i) for i in val[row]])
    # Return the formatted matrix as an array
    return np.asarray(formattedData)


##############################
# GENERATE POLYNOMIAL RANDOM NOISE
##############################
def generate_polynomial_random_noise (degree, coef, num_points):
    """
        Inputs:
            degree (an int): the highest degree in the polynomial
            coef (a list or NumPy array): the length must be the same as 
                degree.  The coefficients for each term in the polynomial.
            num_points (an int): the number of points in the data set.
        Returns:
            x, y (NumPy arrays): the x and y components of the generated 
                polynomial
        Generates a polynomial with random noise for testing purposes.
        The maximum size of the random noise of 5% of the largest generated
        y value.
    """
    # Make sure inputs are all acceptable
    assert isinstance(degree, int)
    assert isinstance(num_points, int)
    assert len(coef) == degree+1
    # Creating the input and output data
    x = np.random.randint(-10, 10, size=(num_points))
    y = np.zeros(num_points)
    for i in np.arange(0, degree+1):
        y = y + coef[i]*x**i
    # Add the noise
    maxy = np.max(y)
    if maxy < 0:
        maxy = -1*maxy
    y_5per = 0.05*maxy
    if y_5per < 1:
        y_5per = 1
    #y = [i + np.random.randint(-y_5per, y_5per) for i in y]
    return x, y     