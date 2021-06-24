##################################################
# Visual Analysis
# Part of the Library: Sequential Regression Extrapolation
# Julie Butler Hartley
# Version 0.0.1
# Date Created: February 20, 2021
# Last Modified: March 1, 2021
#
# A collection of methods to graph data generated by the Sequentual Regression
# Extrapolation library
##################################################

##################################################
# OUTLINE
##################################################

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
import numpy as np 
import matplotlib.pyplot as plt
from Regression import *
from sklearn.metrics import r2_score

class VisualAnalysis():
    ##############################
    # GRAPH MODEL OUTPUTS
    ##############################
    def graph_model_outputs (joint_x_data, y_data, labels, x_label, y_label, 
        savename, isDisplay=True):
        """
            Inputs:
                joint_x_data (a 1D list or NumPy array of numbers): the data to be 
                    plotted on the x axis.  Must be the same of all of the ouputs 
                    given in y_data.
                y_data (a 2D list or NumPy array of numbers): Each row corresponds 
                    to the output of one model.  All rows must be the same lenght 
                    and the same length as joint_x_data.
                labels (a 1D list or NumPy array of strings):  Each string 
                    is the label to be used on the legend for the corresponding
                    data set in the y_data matrix (i.e. the first element in 
                    labels is the legend name for the data in the first row of
                    y_data). 
                x_label (a string): the label for the x axis
                y_label (a string): the label for the y axis
                savename (a string): the file name to save the finished graph to
                isDisplay (a boolean): True case displays the finished graph
            Returns:
                None.

            Plots a series of model outputs from any of the regression methods in 
            this library.
        """
        # Check to see if there is a label for each data set
        assert len(y_data) == len(labels)
        # Make sure the x data is not empty
        assert len(joint_x_data) != 0
        # Make sure the x alvel, the y label, and the file name are all strings
        assert isinstance(x_label, str)
        assert isinstance(y_label, str)
        assert isinstance(savename, str)
        # Make sure the isDisplay variable is a boolean
        assert isinstance(isDisplay, bool)
        # Loop through every data set in y_data
        for i in range(len(y_data)):
            # Make sure its the same length as the x data so there are no graphing
            # problems
            assert len(y_data[i]) == len(joint_x_data)
            # Make sure the label is a string
            assert isinstance(labels[i], str)
            # Plot the current data set with its corresponding label
            plt.plot(joint_x_data, y_data[i], label=labels[i], linewidth=2)
        # Add the x and y labels to the graph.  Also include a legend.
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        # Save the completed graph and if needed display it
        plt.savefig(savename)
        if isDisplay:
            plt.show()

    ##############################
    # VISUALIZE WEIGHTS
    ##############################
    def visualize_weights (weights_matrix, savename, isDisplay=True):
        """
            Inputs:
                weights_matrix (a list): the trained weights from a regression
                    model
                savename (a string): the name for the completed graph to be
                    saved as.
                isDisplay (a boolean): True case displays the completed graph
                    to the screen.  Default value is true
            Returns:
                None.
            Uses the matrix display capabilities of matplotlib to graph the 
            trained weights from a regression algorithm.
        """
        # Check the inputs
        assert isinstance(savename, str)
        assert isinstance(isDisplay, bool)
        # Graph the matrix and add a color bar
        plt.matshow(weights_matrix)
        plt.colorbar()
        # Add labels, save, and display
        plt.xlabel("Weights")
        plt.ylabel("Row")
        plt.savefig(savename)
        if isDisplay:
            plt.show()

    ##############################
    # VISUALIZE 2D DATA
    ##############################  
    def visualize_2d_data (xdata, ydata, label, x_label, y_label, savename,\
        isDisplay=True):
        """
            Inputs:
                xdata (a list or NumPy array): the x component of the data set
                ydata (a list or NumPy array): the y component of the data set
                label (a string): the label of the data set
                x_label (a string): the label for the x axis
                y_label (a string): the label for the y axis
                savename (a string): the name for the completed graph to be
                    saved as.
                isDisplay (a boolean): True case displays the completed graph
                    to the screen.  Default value is true
            Returns:
                None.
            Visualized a 2D data set (i.e. the x component is one dimension
            and the y component is one dimension) using a scatter plot.
        """
        # Check the inputs
        assert isinstance(label, str)
        assert isinstance(x_label, str)
        assert isinstance(y_label, str)
        assert isinstance(savename, str)
        assert isinstance(isDisplay, bool)
        # Make the scatter plot
        plt.scatter(xdata, ydata, label=label)
        # Add axis labels, a legend, save, and display the graph
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(savename)
        if isDisplay:
            plt.show()


    ##############################
    # VISUALIZE 3D DATA
    ##############################  
    def visualize_3d_data (xdata, ydata, x_label, y_label, savename, \
        isDisplay=True):
        """
            Inputs:
                xdata (a 2D list or NumPy array): the x data set.  Must be two
                    dimensions with the first row being the data plotted on 
                    the x axis of the matrix display and the second row being 
                    the data plotted on the y axis of the matrix display.
                ydata (a 2D list of Numpy array): the y component of the data
                    set
                x_label (a string): the label for the x axis
                y_label (a string): the label for the y axis
                savename (a string): the name for the completed graph to be
                    saved as.
                isDisplay (a boolean): True case displays the completed graph
                    to the screen.  Default value is true
            Returns:
                None.
            Visualized a 3D data set (i.e. the x component is two dimensions
            and the y component is one dimension) using a matrix visualization
            from matplotlib.
        """
        # Checking the inputs
        assert len(x_data) == 2
        assert len(y_data)==2
        assert isinstance(x_label, str)
        assert isinstance(y_label, str)
        assert isinstance(savename, str)
        assert isinstance(isDisplay, bool)
        # Create the matrix display and add a color bar
        fig = plt.figure()
        ax = plt.gca()
        fig.matshow(y_data)
        fig.colorbar()
        # Set the tick marks to be the x data
        ax.set_xticks(x_data[0])
        ax.set_yticks(x_data[1])
        # Add x and y labels, save the completed graph and display if needed
        ax.xlabel(x_label)
        ax.ylabel(y_label)
        plt.savefig(savename)
        if isDisplay:
            plt.show()

    ##############################
    # VISUALIZE DIFFERENCE
    ##############################
    def visualize_difference (x_data, true_data, model_data, x_label, y_label,\
        savename, isDisplay=True):
        """
            Inputs:
                x_data  (a list): the x component of the data set
                true_data (a list): the known data set
                model_data (a list): the data set predicted by a regression 
                    algorithm
                x_label (a string): the label for the x axis
                y_label (a string): the label for the y axis
                savename (a string): the name for the completed graph to be
                    saved as.
                isDisplay (a boolean): True case displays the completed graph
                    to the screen.  Default value is true               
            Returns:
                None.
            Plots the difference between the true data set and the predicted
            (model) data set.
        """
        # Check the inputs
        assert len(true_data) == len(model_data)
        assert len(x_data) == len(model_data)
        assert isinstance(x_label, str)
        assert isinstance(y_label, str)
        assert isinstance(savename, str)
        assert isinstance(isDisplay, bool)
        # Find the difference and then plot it
        difference = np.asarray(true_data) - np.asarray(model_data)
        plt.scatter (x_data, difference)
        # Add x and y labels, save the completed graph and display if needed
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(savename)
        if isDisplay:
            plt.show()

    ##############################
    # R2 PLOT
    ##############################
    def R2_plot (true_data, predicted_data, savename, isDisplay=True):
        """
            Inputs:
                true_data (a list or NumPy array): the known data set
                predicted_data (a list or NumPy array): the predicted data set
                    from a Regression algorithm
                savename (a string): the name for the completed graph to be
                    saved as.
                isDisplay (a boolean): True case displays the completed graph
                    to the screen.  Default value is true                               
            Returns:
                None.
            Plots the known data set on the x axis and the predicted data set
            on the y axis.  Adds labels for the R2 score and slope of the 
            graph. Ideally, for a perfect model (i.e. the true data set and
            the predicted data set are exactly the same), both R2 and the 
            slope will be one.
        """
        # Check the inputs
        assert len(true_data) == len(predicted_data)
        assert isinstance(savename, str)
        assert isinstance(isDisplay, bool) 
        # Create a scatter plot of the true and predicted data sets 
        plt.scatter(true_data, predicted_data)
        # Get the R2 score and add it to the plot in the top left corner
        r2 = r2_score(true_data, predicted_data)
        corner1 = np.min(true_data)
        corner2 = np.max(predicted_data)
        plt.annotate("R2 Score: {:.3f}".format(r2), (corner1, corner2), fontsize=16)
        # Get the slope between the two data sets and add it to the plot in
        # the bottom right corner
        lir = LiR(False)
        lir.fit(true_data, predicted_data)
        slope = lir.get_parameters()[0]
        corner3 = np.max(true_data)
        corner4 = np.min(predicted_data)
        plt.annotate("Slope: {:.3f}".format(slope), (corner3, corner4), fontsize=16)
        # Add x and y labels, save the completed graph and display if needed        
        plt.xlabel("True Data")
        plt.ylabel("Predicted Data")
        plt.savefig(savename)
        if isDisplay:
            plt.show()      


##################################################
# Error Analysis
# Part of the Library: Sequential Regression Extrapolation
# Julie Butler Hartley
# Version 0.0.1
# Date Created: February 20, 2021
# Last Modified: March 1, 2021
#
# A collection of methods to graph data generated by the Sequentual Regression
# Extrapolation library
##################################################
class ErrorAnalysis():
    """
        A collection of methods for analyzing the performance of a regression
        algorithm.
    """

    ##############################
    # MSE (Mean-Squared Error)
    ##############################
    def mse (self, A, B):
        """
            Inputs:
                A, B (lists of the same length): two different data sets
            Returns:
                Unnamed (a float): the mean-squared error score between data sets A and B
            Finds the mean-squared error of the two data sets given as inputs.
        """
        A = np.asarray(A)
        B = np.asarray(B)
        return ((A-B)**2).mean()

    ##############################
    # R2
    ##############################
    def r2 (self, true_data, predicted_data):
        """
            Inputs:
                true_data, predicted_data (lists): the known and predicted
                    data sets
            Returns:
                Unnamed (a float): the R2 score between the two data sets
            Calculates and returns the R2 score between a known data set and
            the data set as predicted by a regression algorithm.  For a
            perfect model the R2 score should be 1.
        """
        return r2_score(true_data, predicted_data)

    ##############################
    # ALPHA TUNE
    ##############################
    def alpha_tune (self, X_train, y_train, X_test, y_test, 
            test_alpha_values, isExtrapolate = True, 
            isModified = True, isGraph = False, x_label = '', 
            y_label = '', savename = '', seq=2):
        """
            Inputs:
                X_train, y_train (lists): the training data
                X_test, y_test (lists): the test data set
                test_alpha_values (a list): the alpha value to be tested to
                    find the best one 
                isExtrapolate (a boolean): True case means extrapolation will 
                    be used for prediction, False means the regular prediction
                    method will be used.  Default value is True.
                isModified (a boolean): True means that ridge regression will 
                    normalize the data and set an intercept.  Default case is
                    True
                isGraph (a boolean): True case means the resulting models will
                    be graphed for each tested alpha.  Default value is False.
                x_label (a string): the label for the x axis.  Default value
                    is an empty string.
                y_label (a string): the label for the x axis.  Default value
                    is an empty string.
                savename (a string): the name to save the grapha as. Default
                    value is an empty string
                seq (an int): the length of the sequence used in extrapolation
            Returns:
                best_alpha (a float): the alpha value that give the lowest MSE
                    score
            Tunes a ridge regression model to find the optimal value of alpha
            from a given range.  Works with regular ridge regression or with
            sequential extrapolation format.

        """
        # Check the inputs
        assert len(test_alpha_values) != 0
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert isinstance(isExtrapolate, bool)
        assert isinstance(isModified, bool)
        assert isinstance(isGraph, bool)
        assert isinstance(x_label, str)
        assert isinstance(y_label, str)
        assert isinstance(savename, str)
        assert isinstance(seq, int)
        # Initalize values to determine the best alpha
        best_score = 100
        best_alpha = None
        # If graphing is needed, set up the lists to store the results to be
        # graphed later
        if isGraph:
            models = [y_test]
            scores = []
        # For each value on the list of test values:
        for alpha in test_alpha_values:
            # Create a ridge regression instance
            RR = RR(alpha, isModified)
            # If extrapolation is needed, reformat the training data
            if isExtrapolate:
                X_train, y_train = format_sequential_data(y_train, seq)
            # Fit the RR algorithm and predict the test set using the 
            # method indicated in the inputs
            RR.fit(X_train, y_train)
            if isExtrapolate:
                y_pred = extrapolate(RR, y_train, len(y_test), seq)
            else:
                y_pred = RR.predict(X_test)
            # Get the MSE score and determine if it is the current lowest
            score = mse(y_test, y_pred)
            if mse < best_score:
                best_score = score
                best_alpha = alpha
            # Save the model to be graphed later if needed
            if isGraph:
                models.append(y_pred)
                scores.append(score)
        # Print the best value to the terminal
        print()
        print ("Best alpha value is", best_alpha, "with a model MSE of", best_score)
        print()
        # If graphing is needed:
        if isGraph:
            # Set up the labels, the first is the true data and then one for
            # each of the alpha values, then graph the corresponding data set
            labels = ["True Data"]
            for i in range(len(scores)):
                labels.append("Alpha:" + str(test_alpha_values[i]) + ", Score:" +\
                    str(scores[i]))
            va = VisualAnalysis()
            va.graph_model_outputs (X_test, models, labels, x_label, y_train, 
                savename, isDisplay=True)   
        return best_alpha               

    ##############################
    # KRR PARAMETER TUNE
    ##############################
    def krr_parameter_tune ():
        print("TO BE IMPLEMENTED LATER")
