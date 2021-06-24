##################################################
# Regression
# Part of the Library: Sequential Regression Extrapolation
# Julie Butler Hartley
# Version 0.0.8
# Date Created: February 22, 2021
# Last Modified: February 22, 2021
#
# A series of classes that implement regression methods.  All specific methods
# are children of the main parent class Regression.
# All implementations are closed form analytical solutions which may cause 
# singular decomposition errors with some data sets when taking the inverse
# of the matrices.
##################################################

##################################################
# OUTLINE
##################################################
# Regression
#   The parent class for all of the regression method classes which will
#   be defined later.  Contains all of the common methods that are used
#   by all other regression classes to make the code more mantainable.
#   __init__(self, isModified=True): Initlizer for the parent regression 
#   class.
#   __str__(self): Prints help information to the console if the help method
#   is called on the class.
#   predict(self, points): Uses a trained model to predict values at a set of
#   given points.  Will only work if the model has been train prior to predict
#   being called.
#   get_weights(self): Returns the model weights and prints some useful 
#   information to the console.
#   get_normalization_parameters (self): Returns the constants used to 
#   normalize the data set.  Note that if the class level boolean isModified
#   is False then all of the numbers will be zero.
#   normalize1D (self, X): Normalized only the x component of a data set using
#   pre-defined normalization constants.  Catches if the model is not to be
#   normalized or if the model has not been fit to prevent divide by zero 
#   errors.
#   normalize2D (self, X, y): Normalized a 2D data set using the Standard 
#   Scalar procedure from Scikit-Learn.  This can increase the performance of
#   a regression algorithm.  If the model was not initialized to be normalized,
#   the function just passes the unchanged inputs back.
#   cv_error (self, X, y, k=5): Performs k-fold cross validation on a given 
#   data set to determine the robustness of the algorithm. NOTE: this does not
#   work with sequential regression analysis since that is order depended.
#
# LiR (Linear Regression)
#   Performs linear regression on a data set, including predictions, error
#   analysis, and normalization.  Child class of Regression.
#   __init__(self, isModified=True): Initilizes an instance of the linear
#   regression class.  Sets up all class level variables.  Inherits from 
#   Regression. 
#   __str__(self): Prints help information to the console if the help method
#   is called on the class.  Overwrites inherited function.
#   fit(self, X, y): Trains a linear regression algorithm to find the 
#   optimized weights. Normalizes the data and adds an intercept to be fit if 
#   needed.
#
# RR (Ridge Regression)
#   Performs ridge regression on a data set, including predictions, error 
#   analysis, and normalization.  Child class of Regression.
#   __init__(self, alpha=0.01, isModified=True): Initilizes an instance of the
#   ridge regression class.  Sets up all class level variables. Inherits from
#   Regression.
#   __str__(self): Prints help information to the console if the help method 
#   is called on the class.  Overwrites inherited method.
#   fit(self, X, y): Trains a ridge regression algorithm to find the optimized
#   weights. Normalizes the data and adds an intercept to be fit if needed.
#   
# KRR (Kernel Ridge Regression)
#   Performs kernel ridge regression on a data set, including predictions, 
#   error analysis, and normalization.  Child class of Regression. Note: 
#   overwrites the predict method inherited from Regression.
#   __init__ (self, params, kernel_func, alpha = 0.01, isModified = True):  
#   Initilizes an instance of the kernel ridge regression class.  Sets up all
#   class level variables.  Child class of Regression.
#   __str__(self): Prints help information to the console if the help method 
#   is called on the class.  Overwrites inherited method.
#   polynomial(self, x, y): Polynomial kernel: k(x,y) = (gamma*x*y +c0)^p
#   linear (self, x, y): Linear kernel: k(x,y) = gamma*x*y
#   sigmoid (self, x, y): Sigmoid kernel: k(x,y) = tanh(gamma*x*y + c0)
#   rbf (self, x, y): RBF kernel: k(x,y) = e^(-gamma||x-y||^2)
#   laplacian (self, x, y): Laplacian kernel: k(x,y) = e^(-gamma||x-y||_1)
#   gaussian (self, x, y): Gaussian kernel: k(x,y) = e^(-||x-y||^2/(2sigma^2))
#   modified_gaussian (self, x, y): Modified Gaussian kernel:  
#   k(x,y) = e^(-||x-y||^2/(2sigma^2)) + offset
#   fit (self, X, y): Trains a kernel ridge regression algorithm to find the 
#   optimized weights.
#   predict (self, points): Uses a trained model to predict values at a set of
#   given points. Will only work if the model has been train prior to predict 
#   being called.  Overwrites the inherited method.

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
import numpy as np
# LOCAL IMPORTS
from Support import *
from Extrapolate import *
from Analysis import *

##############################
# REGRESSION (PARENT CLASS)
##############################
class Regression():
    """
        The parent class for all of the regression method classes which will
        be defined later.  Contains all of the common methods that are used
        by all other regression classes to make the code more mantainable.
    """
    # INIT
    def __init__(self, isModified=True):
        """
            Inputs:
                isModified (a boolean): True case means that the data set is
                    normalized prior to use and an intercept is added to the 
                    fit.  Default value is true.
            Returns:
                None.
            Initlizer for the parent regression class.
        """
        # Set up initial values of the class level variables
        self.isModified = isModified
        self.isFit = False
        self.weights = 0.0
        self.isFit = False
        self.X_scale = 0
        self.X_offset = 0
        self.y_offset = 0
        self.y_scale = 0

    # STR
    def __str__(self):
        """
            Inputs:
                None.
            Returns:
                None.
            Prints help information to the console if the help method is
            called on the class  
        """   
        return "Parent Regression Class created by Julie Butler Hartley.  See\
            documentation at ------------."
    
    # PREDICT    
    def predict(self, points):
        """
            Inputs:
                points (a list): the list of points to predicte a value at
            Returns:
                y-hat (a list): a list of the predicted values of the 
                    regression algorithm at each of the given points.  
                    Returns None if fit was not called prior to predict.
            Uses a trained model to predict values at a set of given points.  
            Will only work if the model has been train prior to predict being 
            called.
        """
        # If fit was not called prior to predict (i.e. if the model has not
        # been trained yet)
        if not self.isFit:
            print()
            print ("Model has not been trained so cannot be used to predict.")
            print ("Please call the fit method prior to using the predict\
                    method")
            print()
            return None
        if not isinstance(points, list):
            points = points.tolist()
        if not isinstance(points[0], list):
            print()
            print("Input is 1D list.  Reshaping to 2D list.")
            print()
            points = np.asarray(points).reshape(-1, 1).tolist()
        # Normalize and add the intercept column if needed    
        if self.isModified:
            points = np.hstack((np.ones((len(points),1)), points))
            points = self.normalize1D(points)
        # Find the value of the regression algorithm at each of the given
        # points.  Add the y offset to each predicted value to undo the 
        # normalization.  Note that if isModified is False then y_offset 
        # is zero.
        y_hat = []
        for point in points:
            y_hat.append(point@self.weights)
        y_hat = np.asarray(y_hat).flatten()
        y_hat = self.y_scale*y_hat + self.y_offset
        # Return the predictions
        return y_hat.tolist()    

    # GET PARAMETERS
    def get_weights(self):
        """
            Inputs:
                None.
            Returns:
                self.weights (a list): the weights of the model
            Returns the model weights and prints some useful information to
            the console.

        """
        # Print useful information
        print()
        print("Model has been fitted:", self.isFit)
        print("The weights are:", self.weights)
        print()
        # Return the weights
        return self.weights

    def get_normalization_parameters (self):
        """
            Inputs:
                None.
            Returns:
                y_offset (a float): the average of the y data used to train
                    the model
                X_offset (a float): the average of the x data used to train
                    the model
                X_scale (a float): the L2 norm of the x data used to train
                    the model
            Returns the constants used to normalize the data set.  Note that 
            if the class level boolean isModified is False then all of the 
            numbers will be zero.    
        """
        print()
        print("Model has been fitted:", self.isFit)
        print("Y offset:", self.y_offset)
        print("X offset:", self.X_offset)
        print("X scale:", self.X_scale)
        print()
        return self.y_offset, self.y_scale, self.X_offset, self.X_scale   

    # NORMALIZE 1D
    def normalize1D (self, X):
        """
            Inputs:
                X (a NumPy array): a set of x data to be normalized
            Returns:
                Unnamed (a NumPy array): the x data after standard 
                    normalization is applied with constants found in the
                    fit method.  Returns None if the model has not been fit or
                    if the model is not set up for normalization.
            Normalized only the x component of a data set using pre-defined 
            normalization constants.  Catches if the model is not to be
            normalized or if the model has not been fit to prevent divide by
            zero errors.
        """
        if self.isFit and self.isModified:
            return (X-self.X_offset)/self.X_scale
        else:
            print()
            print("Model has not been fit or is not set up to normalize the\
                    data.  Returning None.")
            print()
            return None

    # NORMALIZE 2D
    def normalize2D (self, X, y):
        """
            Inputs:
                X,y (NumPy arrays): the x and y components of the data set to 
                    be normalized.
            Returns:
                X,y (NumPy arrays): the data after being normalized using the 
                    Standard Scalar procedure from Scikit-Learn.  If the model
                    was not initialized to be normalized then the function 
                    simply returns the inputs (unchanged).
            Normalized a 2D data set using the Standard Scalar procedure from
            Scikit-Learn.  This can increase the performance of a regression
            algorithm.  If the model was not initialized to be normalized, the
            function just passes the unchanged inputs back.
        """
        # To normalize the data:
        if self.isModified:
            # Subtract the average of the y data from the y data
            self.y_offset = np.average(y)
            self.y_scale = np.linalg.norm(y)
            y = (y - self.y_offset)/self.y_scale
            # Subtract the average of the X data from the X data and then 
            # divide by the L2 norm of the X data
            self.X_offset = np.average(X)
            X = X - self.X_offset
            self.X_scale = np.linalg.norm(X)
            X = X/self.X_scale
            return X, y  
        else:
            print()
            print("Model is not to be normalized.  This option is set when\
                    initilizing the class.  Returning unmodified data,")
            print() 
            return X,y  

    # BOOTSTRAP ERROR
    def bootstrap_error (self):
        print ("BOOTSTRAP ERROR TO BE IMPLEMENTED LATER")

    # CV (CROSS VALIDATION) ERROR
    def cv_error (self, X, y, k=5):
        """
            Inputs:
                X,y (lists or NumPy arrays): the data set to be used for cross
                    validation
                k (an int): the number of folds to be used
            Returns:
                cv_error (a float): the cross validation error
                errors (a list of floats): the MSE error from each fold
                weights (a list of lists): the trained weights from each fold
            Performs k-fold cross validation on a given data set to determine
            the robustness of the algorithm.
            NOTE: this does not work with sequential regression analysis since 
            that is order depended.
        """
        assert len(X) == len(y)
        assert isinstance(k, int)
        # Calculate the number of points in each fold
        length_fold = int(len(y)/k)
        # Set up lists to hold data
        X_partitioned = []
        y_partitioned = []
        errors = []
        weights = []
        EA = ErrorAnalysis()
        # Partition the data set into k pieces
        for f in range(k):
            start_index = f*length_fold
            end_index = (f+1)*length_fold
            X_partitioned.append(X[start_index:end_index])
            y_partitioned.append(y[start_index:end_index])
        # Use each partiion as the test data and the remainder of the data set
        # as the training data
        for f in range(k):
            # Get the test data
            X_test = X_partitioned[f]
            y_test = y_partitioned[f]
            # Get the training data
            X_train = [x for i,x in enumerate(X_partitioned) if i!=f]
            y_train = [x for i,x in enumerate(y_partitioned) if i!=f]
            # Fit the model with the training data and then predict the values
            # of the test set
            self.fit(X_train, y_train)
            y_predict = self.predict(X_test)
            # Calculate the error in the test set
            error = EA.mse(y_predict, y_test)
            # Save the weights and error
            weights.append(self.get_weights())
            errors.append(error)
        # Calcualte the cv error, the average of the errors from each fold
        cv_error = np.average(errors)
        return cv_error, errors, weights    

##############################
# LiR (LINEAR REGRESSION)
##############################
class LiR(Regression):
    """
        Performs linear regression on a data set, including predictions, error
        analysis, and normalization.  Child class of Regression.
    """
    # INIT
    def __init__(self, isModified=True):
        """
            Inputs:
                isModified (a boolean): True case means modifed linear
                    regression will be used, meaning the data will be 
                    normalized and an intercept will be used.
            Returns:
                None.
            Initilizes an instance of the linear regression class.  Sets up
            all class level variables.  Inherits from Regression.             
        """
        # Initilize the parent class
        Regression.__init__(self, isModified)
        print()
        print("Starting Linear Regression")
        print()

    # STR
    def __str__(self):
        """
            Inputs:
                None.
            Returns:
                None.
            Prints help information to the console if the help method is
            called on the class.  Overwrites inherited function.    
        """
        return "Linear Regression Class created by Julie Butler Hartley.  See\
            documentation at ------------."

    # FIT
    def fit(self, X, y):
        """
            Inputs:
                X, y (lists or Numpy arrays): the x and y components of the
                    data set
            Returns:
                None.
            Trains a linear regression algorithm to find the optimized weights.
            Normalizes the data and adds an intercept to be fit if needed.
        """
        if not isinstance(X[0], list):
            print(X[0])
            print()
            print("Input is 1D list.  Reshaping to 2D list.")
            print()
            X = np.asarray(X).reshape(-1, 1).tolist()
        # If the model is initialzed as modified then add a column of ones so
        # the intercept will be found and normalize the data.
        if self.isModified:
            print("Performing Modified Linear Regression")
            print("Data Normalization and Fitting an Intercept Will Be Used")
            # Add the column of ones and normalize (normalization method 
            # inherited from Regression)
            X = np.hstack((np.ones((len(X),1)), X))
            X, y = self.normalize2D(X, y)
        # If the model is not modified
        else:
            print("Performing Naive Linear Regression")   
        # Find the trained weights using the analytical, closed-form expression
        weights = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
        self.weights = np.asarray(weights).flatten()
        # Update the model to know that it has been fit
        self.isFit = True
         
    # ANALYTICAL ERROR
    def analytical_error(self,X,y):
        sigma_squared = np.var(y)
        variance = sigma_squared*np.linalg.inv(X.T@X)
        parameter_variance = np.diagonals(variance)
        print("Analytical error implementation to be finished")


##############################
# RR (Ridge Regression)
##############################
class RR(Regression):
    """
        Performs ridge regression on a data set, including predictions, error
        analysis, and normalization.  Child class of Regression.
    """
    # INIT
    def __init__(self, alpha=0.01, isModified=True):
        """
            Inputs:
                isModified (a boolean): True case means modifed linear
                    regression will be used, meaning the data will be 
                    normalized and an intercept will be used.
            Returns:
                None.
            Initilizes an instance of the ridge regression class.  Sets up
            all class level variables.  Child class of Regression.            
        """
        # alpha belongs only to the ridge regression class
        self.alpha = alpha
        Regression.__init__(self, isModified)
        print("Starting Linear Regression")

    # STR
    def __str__(self):
        """
            Inputs:
                None.
            Returns:
                None.
            Prints help information to the console if the help method is
            called on the class.  Overwrites inherited method.  
        """
        return "Ridge Regression Class created by Julie Butler Hartley.  See\
            documentation at ------------."
 
    # FIT
    def fit(self, X, y):
        """
            Inputs:
                X, y (lists or Numpy arrays): the x and y components of the
                    data set
            Returns:
                None.
            Trains a ridge regression algorithm to find the optimized weights.
            Normalizes the data and adds an intercept to be fit if needed.
        """
        if not isinstance(X[0], list):
            print()
            print("Input is 1D list.  Reshaping to 2D list.")
            print()
            X = np.asarray(X).reshape(-1, 1).tolist()
        print ("Solving Using Closed Form Ridge Regression")
        print ("Iterative Solving Methods May be Implemented in later versions")
        # If the model is initialzed as modified then add a column of ones so
        # the intercept will be found and normalize the data.        
        if self.isModified:
            print("Performing Modified Ridge Regression")
            print("Data Normalization and Fitting an Intercept Will Be Used")
            # Add the column of ones and normalize (normalization method 
            # inherited from Regression)            
            X = np.hstack((np.ones((len(X),1)), X))
            X, y = self.normalize2D(X, y)
        # If the model is not modified
        else:
            print("Performing Naive Ridge Regression") 
        # Find the trained weights using the analytical, closed-form expression
        rows, cols = np.asarray(X).shape       
        weights = np.linalg.inv(np.transpose(X)@X - self.alpha*np.eye(cols))@np.transpose(X)@y
        self.weights = np.asarray(weights).flatten()
        # Update the model to know that it has been fit
        self.isFit = True

    # ANALYTICAL ERROR
    def analytical_error (self):
        print ("ANALYTICAL ERROR TO BE IMPLMENTED LATER")       


##############################
# KRR (Kernel Ridge Regression)
##############################
class KRR (Regression):
    """
        Performs kernel ridge regression on a data set, including predictions, 
        error analysis, and normalization.  Child class of Regression.  
        Note: overwrites the predict method inherited from Regression.
    """

    # INIT
    def __init__ (self, params, kernel_func, alpha = 0.01, isModified = True):
        """ 
            Inputs:
                params (a list): the parameters needed by the kernel function
                kernel_func (a string): a string that corresponds to the
                    kernel function to be used.
                isModified (a boolean): True case means modifed linear
                    regression will be used, meaning the data will be 
                    normalized and an intercept will be used.
            Returns:
                None.
            Initilizes an instance of the kernel ridge regression class.  Sets
            up all class level variables.  Child class of Regression.  
        """
        print()
        print("Starting Kernel Ridge Regression")
        print()
        # Set up class level variables
        self.params = params
        self.X_train = 0
        self.alpha = alpha
        # Set up the kernel function using the given string.  If the strong is
        # not recognized then end the program.
        if kernel_func == "polynomial" or kernel_func == 'p':
            print()
            print("Setting kernel function to polynomial.")
            print()
            assert len(params) == 3
            self.kernel_func = self.polynomial
        elif kernel_func == "linear" or kernel_func == 'l':
            print()
            print("Setting kernel function to linear.")
            print()
            assert len(params) == 1
            self.kernel_func = self.linear
        elif kernel_func == "sigmoid" or kernel_func == 's':
            print()
            print("Setting kernel function to sigmoid.")
            print()
            assert len(params) == 2
            self.kernel_func = self.sigmoid
        elif kernel_func == "rbf" or kernel_func == 'r':
            print()
            print("Setting kernel function to radial basis function.")
            print()
            assert len(params) == 1
            self.kernel_func = self.rbf
        elif kernel_func == "laplacian" or kernel_func == 'l':
            print()
            print("Setting kernel function to Laplacian.")
            print()
            assert len(params) == 1
            self.kernel_func = self.laplacian
        elif kernel_func == "gaussian" or kernel_func == 'g':
            print()
            print("Setting kernel function to Gaussian.")
            print()
            assert len(params) == 1
            self.kernel_func = self.gaussian 
        elif kernel_func == "modified gaussian" or kernel_func == 'm':
            print()
            print("Setting kernel function to modified Gaussian.")
            print()
            assert len(params) == 2
            self.kernel_func = self.modified_gaussian  
        # End the program if the kernel name is not recognized        
        else:
            print ("Invalid Kernel Function Name") 
            print ("Valid Kernel Names: polynomial, linear, sigmoid, rbf,\
                     laplacian, gaussian")
            print ("Please Initialize with a Valid Kernel Name") 
            print ("Program will not end.")
            import sys
            sys.exit()  
        # Setting up the parent class
        Regression.__init__(self, isModified)

    # STR
    def __str__(self):
        """
            Inputs:
                None.
            Returns:
                None.
            Prints help information to the console if the help method is
            called on the class.  Overwrites inherited method. 
        """
        return "Kernel Ridge Regression Class created by Julie Butler Hartley.\
            See documentation at ------------."    

    ##############################
    # KERNELS
    # Inputs for all kernels:
    #   x,y (NumPy arrays or floats): inputs to the kernel function
    # Returns for all kernels:
    #   k (Numpy array or float): the value of the kernel function at x,y
    ##############################
    # POLYNOMIAL
    def polynomial(self, x, y):
        """
            Polynomial kernel: k(x,y) = (gamma*x*y +c0)^p
        """
        gamma = self.params[0]
        c0 = self.params[1]
        p = self.params[2]
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        k = (gamma*np.dot(x,y) +c0)**p
        return k.tolist()

    # LINEAR
    def linear (self, x, y):
        """
            Linear kernel: k(x,y) = gamma*x*y
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        return gamma*np.dot(x,y).tolist()

    # SIGMOID
    def sigmoid (self, x, y):
        """
            Sigmoid kernel: k(x,y) = tanh(gamma*x*y + c0)
        """
        gamma = self.params[0]
        c0 = self.params[1]
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()        
        k = np.tanh(gamma*np.dot(x,y)+c0)
        return k.tolist()

    # RBF (Radial Basis Function)
    def rbf (self, x, y):
        """
            RBF kernel: k(x,y) = e^(-gamma||x-y||^2)
        """
        gamma = self.params[0]
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()        
        k = np.exp(-1*gamma*np.norm(x-y))
        return k.tolist()

    # LAPLACIAN
    def laplacian (self, x, y):
        """
            Laplacian kernel: k(x,y) = e^(-gamma||x-y||_1)
        """
        gamma = self.params[0]
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()        
        k = np.exp(-1*gamma*np.norm(x-y, 1))
        return k.tolist()

    # GAUSSIAN
    def gaussian (self, x, y):
        """
            Gaussian kernel: k(x,y) = e^(-||x-y||^2/(2sigma^2))
        """
        sigma = params[0]
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()        
        k = np.exp(-1*np.norm(x-y)/(2*sigma^2))
        return k.tolist()

    # MODIFIED GAUSSIAN
    def modified_gaussian (self, x, y):
        """
            Modified Gaussian kernel: 
            k(x,y) = e^(-||x-y||^2/(2sigma^2)) + offset
        """
        sigma = params[0]
        offset = params[1]
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()        
        k = np.exp(-1*np.norm(x-y)/(2*sigma^2)) + offset
        return k.tolist()

    ##############################
    # OTHER FUNCTIONS
    ##############################
    # FIT
    def fit (self, X, y):
        """
            Inputs:
                X, y (lists or Numpy arrays): the x and y components of the
                    data set
            Returns:
                None.
            Trains a kernel ridge regression algorithm to find the 
            optimized weights.
        """
        # Create the kernel matrix, K
        kernel = []
        for i in range(len(X)):
            row = []
            for j in range(len(X)):
                row.append(self.kernel_func(X[i], X[j]))
            kernel.append(row)
        kernel = np.array(kernel)
        # Get the size of the kernel matrix
        id_size = len(kernel)
        # Find the trained weights using the analytical, closed-form expression       
        self.weights = np.linalg.inv(kernel + self.alpha*np.identity(id_size))@y
        # Update the model to know that it has been fit
        self.isFit = True
        # Save the training data which will be needed in the predict function
        self.X_train = X

    # PREDICT
    def predict (self, points):
        """
            Inputs:
                points (a list): the list of points to predicte a value at
            Returns:
                y-hat (a list): a list of the predicted values of the 
                    regression algorithm at each of the given points.  
                    Returns None if fit was not called prior to predict.
            Uses a trained model to predict values at a set of given points.  
            Will only work if the model has been train prior to predict being 
            called.  Overwrites the inherited method.
        """    
        # If fit was not called prior to predict (i.e. if the model has not
        # been trained yet)
        if not self.isFit:
            print()
            print ("Model has not been trained so cannot be used to predict.")
            print ("Please call the fit method prior to using the predict\
                    method")
            print()
            return None
        else:
            # Predict the value at each of the given points using the trained KRR algorithm
            y_hat = []
            for x in range(len(points)):
                pred = 0
                for i in range(len(self.X_train)):
                    pred += self.weights[i]*self.kernel_func(self.X_train[i],points[x])
                y_hat.append(pred)
            return y_hat   