Problem Statement
------------------

1. This course project simulates the application of Reinforcement Learning to a real problem, like a medical application or business application, where deploying a policy worse than the current one would be dangerous or costly.

Directory structure
--------------------

1. We have the data of 200,000 episodes in the file "data.txt" present within the "data" folder, but not the underlying MDP that created the data.
2. The "output" folder has the 100 newly generated files. Each file contains one row corresponding to new policy parameters to use in place of theta_b using the same policy parameterization described for pie_b.

Instructions to run the program
-------------------------------

1. From the "source" folder", run the "main.py" file in Python.
2. The Python version supported is 3.7.
3. Type the command "python main.py" in the command line to start the execution of the program.
4. After the program completes execution, the 100 newly generated files containing new policy parameters are stored in the "output" folder and also in the directory which is the parent of the "source" directory as .csv files.