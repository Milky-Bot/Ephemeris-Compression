import numpy as np
import numpy.core.numeric as NX
from numpy.core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
                        ones)
from numpy.linalg import eigvals, lstsq, inv
class elm():
    '''
    ELM class:
    Extreme Learning Machine for regression
    '''
    
    def __init__(self, hidden_units,  x, y, w=None, weight_seed=None, bias_seed=None):
        '''
        Function: _init
            initialize input ELM

        INPUTS :
            hidden_units (INTEGER): number of neurons in the hidden layer
            x (ARRAY of shape[samples, features]): input data points
            y: (ARRAY of shape[samples,]): target values
                
            weights_seed (INTEGER): random weights for generating input weights, default:None
            bias_seed (INTEGER): random weights for generating input biases, default:None
        '''
        
        self.hidden_units = hidden_units
        self.x = x
        self.y = y
        self.weight_seed = weight_seed
        self.bias_seed = bias_seed
        self.class_num = self.y.shape[1]  
        self.beta = np.zeros((self.hidden_units, self.class_num))   
        self.w=w


        np.random.seed(self.weight_seed)
        self.W = np.random.normal(loc=0, scale=self.hidden_units//2, size=(self.hidden_units, self.x.shape[1]))
        np.random.seed(self.bias_seed)
        self.b = np.random.normal(loc=0, scale=self.hidden_units//2, size=(self.hidden_units, 1))
        

    
    def in_to_hidden(self, x):
        '''
        compute the matrix H 
        '''
        
        self.Htemp = np.dot(self.W, x.T) + self.b
        self.H = np.sin(self.Htemp)
        return self.H.T

    def fit(self):
        '''
        Function fit: train the ELM, computing the matrix of weights going from hidden layer to output layer
        
        OUTPUT:
            beta (ARRAY): matrix of weights going from hidden layer to output layer
        '''
        
        self.H_temp = self.in_to_hidden(self.x)
        scale = NX.sqrt((self.H_temp *self.H_temp).sum(axis=0))
        self.H_temp /= scale
        
        rhs=self.y.reshape(-1)
        if self.w is not None:
            self.w = NX.asarray(self.w) + 0.0
            c, resids, rank, s = lstsq(self.H_temp* self.w[:, NX.newaxis], rhs*self.w, rcond=None)
            
        else:
             c, resids, rank, s = lstsq(self.H_temp, rhs, rcond=None)
        self.beta = (c.T/scale).T  # broadcast scale coefficients
        
        return self.beta,self.W,self.b
    

    def hidden_to_out(self, H):
        '''
        compute output of the elm
        '''
        self.output = np.dot(H, self.beta)
        return self.output


    def predict(self, x):
        '''
        Function: compute ELM output, provided some values x
        
        INPUT:
            x (ARRAY, shape[samples, features]) inputs to the ELM
        Return:
            y_(ARRAY, shape[samples,]): predictions
        '''
            
        self.H = self.in_to_hidden(x)
        self.y_ = self.hidden_to_out(self.H)
        return self.y_
