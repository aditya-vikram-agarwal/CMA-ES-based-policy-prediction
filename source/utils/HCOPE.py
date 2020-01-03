import numpy as np
import random
from scipy.stats import t
import cma
import cma.test

from .FourierBasisPolicy import FourierBasisSoftmax


class HCOPE(object):
    
    def __init__(self, D, num_Ds, c, pi_b, delta, gamma, sigma=0.5):
        
        self.D = D
        self.num_Ds = num_Ds
        self.c = c
        self.pi_b = pi_b
        self.delta = delta
        self.gamma = gamma
        self.sigma = sigma
        self.barrier = 0.0
        
        self.Ds, self.Dc = HCOPE.data_partition(self.D, self.num_Ds)

        self.episodeCount = len(D)
        
        self.cma_function = cma_optimizer(self.Dc, self.num_Ds, self.c, self.pi_b, self.delta, self.barrier)
        self.es = cma.CMAEvolutionStrategy(self.pi_b.parameters.copy(), self.sigma)
        
    @staticmethod
    def data_partition(D, num_Ds):
        
        return D[: num_Ds], D[num_Ds: ]

    @staticmethod
    def pdis_for_data(D_data, pi_e, pi_b, gamma = 1.0):
        
        pdis = 0.0

        n = len(D_data)
        
        for i in range(n):
            pdis += HCOPE.pdis_for_history(D_data[i], pi_e, pi_b)
        
        pdis /= n
        
        return pdis

    @staticmethod
    def pdis_for_history(H, pi_e, pi_b, gamma = 1.0):
        
        L = H.episodeLength
        return_value = 0.0
        
        for t in range(L):
            
            p_ie_p_ib = 1.0
            R_t = H.rewards[t]
            
            for j in range(t + 1):
                
                S_j = H.states[j]
                A_j = H.actions[j]
                p_ie_p_ib *= (pi_e(S_j, A_j)) / (pi_b(S_j, A_j))
                
            return_value += (p_ie_p_ib * R_t)
            
        return return_value

    @staticmethod
    def variance_D(D, pi_e, pi_b):
        
        n = len(D)
        
        average_pdis = HCOPE.pdis_for_data(D, pi_e, pi_b)

        total_pdis_square = 0.0
        
        for i in range(n):
            H_i = D[i]
            total_pdis_square += ((HCOPE.pdis_for_history(H_i, pi_e, pi_b) - average_pdis) ** 2)

        total_pdis_square = np.sqrt(total_pdis_square / (n - 1))

        return total_pdis_square


    @staticmethod
    def candidate_test(Dc, pi_e, pi_b, num_Ds, delta, c):
        
        PDIS_Dc_pi_e_pi_b = HCOPE.pdis_for_data(Dc, pi_e, pi_b)

        variance_c = HCOPE.variance_D(Dc, pi_e, pi_b)

        safety = PDIS_Dc_pi_e_pi_b - (2 * (variance_c / np.sqrt(num_Ds)) * t.ppf(1 - delta, num_Ds - 1))

        print("candidate theta variance: " + str(variance_c))

        return (safety >= c), - PDIS_Dc_pi_e_pi_b


    @staticmethod
    def safety_test(Ds, pi_e, pi_b, num_Ds, delta, c):
        
        PDIS_Ds_pi_e_pi_b = HCOPE.pdis_for_data(Ds, pi_e, pi_b)

        variance_s = HCOPE.variance_D(Ds, pi_e, pi_b)

        safety = PDIS_Ds_pi_e_pi_b - ((variance_s / np.sqrt(num_Ds)) * t.ppf(1 - delta, num_Ds - 1))

        print("safety variance : " + str(variance_s))

        return (safety >= c), PDIS_Ds_pi_e_pi_b


    def get_new_policy(self, Dc):
        
        possible_theta, evaluation_results = self.es.ask_and_eval(self.cma_function)
        min_index = np.argmin(evaluation_results)
        
        print("evaluation_results: " + str(evaluation_results))
        
        self.es.tell(possible_theta, evaluation_results)
        self.es.logger.add()
        
        candidate_theta = possible_theta[min_index]
        
        
        #safety check#
        
        safety_pass = False
        if evaluation_results[min_index] < 100:
            print("----Candidate theta found-----")
            print("::::::::::::::::::::Checking Safety Test:::::::::::::")
            numStates = self.pi_b._numStates
            numActions = self.pi_b._numActions
            iOrder = self.pi_b._iOrder
            dOrder = self.pi_b._dOrder

            pi_e = FourierBasisSoftmax(numStates, numActions, iOrder, dOrder)
            pi_e.parameters = candidate_theta

            safety_pass, safety_pdis = HCOPE.safety_test(self.Ds, pi_e, self.pi_b, self.num_Ds, self.delta, self.c)
            
            print("Safety_pdis: " + str(safety_pdis))
        ##

        return candidate_theta, evaluation_results[min_index], safety_pass

    def get_new_candidate_theta(self):
        
        candidate_theta, result, safety_pass = self.get_new_policy(self.Dc)
        
        return candidate_theta, result, safety_pass


class cma_optimizer(object):
    
    def __init__(self, Dc, num_Ds, c, pi_b, delta, barrier):
        
        self.Dc = Dc
        self.num_Ds = num_Ds
        self.c = c
        self.pi_b = pi_b
        self.delta = delta
        self.barrier = barrier

    def __call__(self, theta):
        
        numStates = self.pi_b._numStates
        numActions = self.pi_b._numActions
        iOrder = self.pi_b._iOrder
        dOrder = self.pi_b._dOrder
        
        pi_e = FourierBasisSoftmax(numStates, numActions, iOrder, dOrder)
        pi_e.parameters = theta
        
        candidate_pass, x = HCOPE.candidate_test(self.Dc, pi_e, self.pi_b, self.num_Ds, self.delta, self.c)
        
        print(x)
        print("candidate_pass: " + str(candidate_pass))
              
        if candidate_pass == True:
              return x
        else:
            return  x + 1000000