'''
Name: Jacob Sherlin
Due Date: 21 October 2024
Course: CSCI-4350-001
Project: OLA #2
Project Description: Use greedy local search (gradient ascent) to obtain the maximum value of the SoG function.

A.I. Disclaimer: All work for this assignment was completed by myself and
entirely without the use of artificial intelligence tools such as ChatGPT, MS
Copilot, other LLMs, etc.

'''

import numpy as np
import sys
from SumofGaussians import SumofGaussians

def main():
    #read command line arguments
    random_seed = int(sys.argv[1])  #random seed
    dimensions = int(sys.argv[2])  #number of dimensions
    num_centers = int(sys.argv[3])  #number of Gaussians

    #initialize random number generator and sog function
    rng = np.random.default_rng(random_seed)
    sog = SumofGaussians(dimensions, num_centers, rng)

    #constants
    tolerance = 1e-8
    max_iters = 100000
    learning_rate = 0.01

    #start at a random point in the [0, 10]^d cube
    point = rng.uniform(0, 10, size=dimensions)

    #gradient ascent loop
    for iteration in range(max_iters):
        current_value = sog.Evaluate(point)
        gradient_value = sog.Gradient(point)

        #print current point and function value
        display_current_state(point, current_value)

        #move in the direction of the gradient
        new_point = point + learning_rate * gradient_value
        new_value = sog.Evaluate(new_point)

        #termination check
        if is_termination_criteria_met(current_value, new_value, tolerance):
            break

        #update current point
        point = new_point

    #final print for the last iteration
    display_current_state(point, new_value)

def display_current_state(point, current_value):
    #print the current state of point and current_value formatted to 8 decimal places
    formatted_coords = " ".join(f"{coord:.8f}" for coord in point)
    print(f"{formatted_coords} {current_value:.8f}")

def is_termination_criteria_met(current_value, new_value, tolerance):
    #check if need for termination
    return abs(new_value - current_value) < tolerance

if __name__ == "__main__":
    main()
