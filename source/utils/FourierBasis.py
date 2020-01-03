import numpy as np

import math

class FourierBasis():
    
    def __init__(self, inputDimension, iOrder, dOrder):
        
        self.inputDimension = inputDimension
        iTerms = iOrder * inputDimension
        dTerms = np.power(dOrder + 1, inputDimension)
        oTerms = min(iOrder, dOrder) * inputDimension
        self.nTerms = iTerms + dTerms - oTerms
        self.c = np.zeros((self.nTerms, inputDimension))
        counter = np.zeros(inputDimension)
        
        for termCount in range(dTerms):
            
            self.c[termCount, :] = counter
            self.incrementCounter(counter, dOrder)
            
        termCount = dTerms
        
        for i in range(inputDimension):
            for j in range(dOrder + 1, iOrder + 1):
                
                self.c[termCount, :] = np.zeros(inputDimension)
                self.c[termCount, i] = j
                termCount = termCount + 1

    def incrementCounter(self, counter, dOrder):
        
        for i in range(len(counter)):
            
            counter[i] = counter[i] + 1
            
            if counter[i] > dOrder:
                counter[i] = 0
            else:
                break

    def getNumOutputs(self):
        
        return self.nTerms

    def basify(self, x):
        
        result = np.zeros(self.nTerms)
        
        for i in range(self.nTerms):
            result[i] = math.cos(math.pi * np.dot(self.c[i], x))
            
        return result

    def incrementCounter(self, counter, dOrder):
        
        for i in range(len(counter)):
            
            counter[i] = counter[i] + 1
            
            if counter[i] > dOrder:
                counter[i] = 0
            else:
                break