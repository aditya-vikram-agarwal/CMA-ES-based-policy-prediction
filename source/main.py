import numpy as np
import random

from utils.load_and_output_data_lib import load_data_from_file, output_data_to_file

from utils.HCOPE import HCOPE
from utils.FourierBasisPolicy import FourierBasisSoftmax

num_state_features, num_discrete_actions, fourier_basis_order, theta_b, num_episodes, D, pi_b_St_At = load_data_from_file("data/data.csv")

print("--------------------------------Printing extracted values from data.txt--------------------------------")

print("Number of state features: " + str(num_state_features))
print("Number of discrete actions: " + str(num_discrete_actions))
print("Fourier basis order: " + str(fourier_basis_order))
print("Theta b: " + str(theta_b))
print("Number of episodes: " + str(num_episodes))
print("Printing states for first 5 episodes (extracted states for all " + str(len(D)) + " episodes): " + str(D[0].states) + str(D[1].states) + str(D[2].states) + str(D[3].states) + str(D[4].states))
print("Printing actions for first 5 episodes (extracted actions for all " + str(len(D)) + " episodes): " + str(D[0].actions) + str(D[1].actions) + str(D[2].actions) + str(D[3].actions) + str(D[4].actions))
print("Printing rewards for first 5 episodes (extracted rewards for all " + str(len(D)) + " episodes): " + str(D[0].rewards) + str(D[1].rewards) + str(D[2].rewards) + str(D[3].rewards) + str(D[4].rewards))
print("Pi_b(St, At) : " + str(pi_b_St_At))

print("---------------------------------Extracted all data from data.txt----------------------------------------")

J_pi_b = 0.0
for H in D:
    J_pi_b += np.sum(H.rewards)

J_pi_b /= len(D)

print("J_pi_b: " + str(J_pi_b))


random.shuffle(D)


J_pi_b = 0.0
for H in D:
    J_pi_b += np.sum(H.rewards)

J_pi_b /= len(D)

print("Sample J_pi_b: " + str(J_pi_b))


Ds_size = int(0.5 * len(D))
delta = 0.05
gamma = 1.0
c = 2.0

pi_b = FourierBasisSoftmax(num_state_features, num_discrete_actions, fourier_basis_order, 0)
pi_b.parameters = theta_b

print(pi_b.parameters)
print("-------------------------------")

hcope_object = HCOPE(D, Ds_size, c, pi_b, delta, gamma)

results = []
    
file_number = 0

while file_number < 100:

    candidate_theta, result, safety_pass = hcope_object.get_new_candidate_theta()

    print("return: " + str(result) + " theta: " + str(candidate_theta))
    print("safety_test_pass: " + str(safety_pass))
    
    if safety_pass == True:
        results.append([result, candidate_theta])
        print("Outputting " + str(candidate_theta) + " " + str(result) + " to .csv file")
        output_data_to_file("output/" + str(file_number + 1) + ".csv", candidate_theta)
        output_data_to_file("output/" + "returns_" + str(file_number + 1) + ".csv", [result])
        output_data_to_file("../" + str(file_number + 1) + ".csv", candidate_theta)
        output_data_to_file("../" + "returns_" + str(file_number + 1) + ".csv", [result])
        file_number += 1
    print("Number of theta obtained: " + str(file_number))
        
print(results)