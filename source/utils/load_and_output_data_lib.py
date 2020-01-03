import numpy as np
import csv

class Episode(object):
    
    def __init__(self, states, actions, rewards):
        
        self.states = states
        self.actions = actions
        self.rewards = rewards
        
        self.episodeLength = len(actions)

def load_data_from_file(relative_file_path):

    num_state_features = None
    num_discrete_actions = None
    fourier_basis_order = None
    theta_b = None
    num_episodes = None
    T = []
    pi_b_St_At = None

    with open(relative_file_path) as file:
        
        new_list = [new_line.split() for new_line in file] 
        
        for element_number, data in enumerate(new_list): 
            
            if element_number == 0:
                num_state_features = int(data[0])
                
            elif element_number == 1:
                num_discrete_actions = int(data[0])
                
            elif element_number == 2:
                fourier_basis_order = int(data[0])
                
            elif element_number == 3:
                theta_b = np.fromstring(data[0], dtype=float, sep=',')
                
            elif element_number == 4:
                num_episodes = int(data[0])
                
            elif element_number > 4 and element_number < 5 + num_episodes:
                
                string_format = np.fromstring(data[0], dtype=float, sep=',')
                
                states = []
                actions = []
                rewards = []
                
                for index in range(len(string_format)):
                    
                    if index % 3 == 0:
                        states.append(string_format[index])
                    elif index % 3 == 1:
                        actions.append(string_format[index])
                    else:
                        rewards.append(string_format[index])
                        
                episode = Episode(states, actions, rewards)
                T.append(episode)
                        
            else:
                pi_b_St_At = np.array(np.fromstring(data[0], dtype=float, sep=','))
                
    return num_state_features, num_discrete_actions, fourier_basis_order, theta_b, num_episodes, T, pi_b_St_At

def output_data_to_file(relative_file_path, data):
    
    with open(relative_file_path, "w") as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerow(data)