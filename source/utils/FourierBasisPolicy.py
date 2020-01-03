import numpy as np
from typing import Union

from .FourierBasis import FourierBasis
from .skeleton import Policy

np.random.seed(1000007)


class FourierBasisSoftmax(Policy):
    
    def __init__(self, numStates : int, numActions : int, iOrder : int, dOrder : int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        
        #TODO
        self._numStates = numStates
        self._numActions = numActions
        self._iOrder = iOrder
        self._dOrder = dOrder
        self._fourier_basis = FourierBasis(self._numStates, self._iOrder, self._dOrder)
        self._numFeat = self._fourier_basis.getNumOutputs()
        self._theta = np.zeros((self._numActions, self._numFeat))
        self._phi = None
        self._p = None


    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        
        #TODO
        if action is not None:
            return self.getActionProbabilities(state)[int(action)]
        else:
            return self.getActionProbabilities(state)

    def samplAction(self, state:np.ndarray)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        
        #TODO
        
        return np.random.choice(np.arange(self._numActions), p = self.getActionProbabilities(state))
    
    def calculate_phi(self, state):

        return self._fourier_basis.basify(state)

    def getActionProbabilities(self, state:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """

        #TODO
        
        self._phi = self.calculate_phi(state)
        
        x = np.dot(self._theta, self._phi)
        self._p = x - np.max(x)
        
        exp_state_row = np.exp(self._p)
        
        probabilities = exp_state_row / np.sum(exp_state_row)
        
        return probabilities