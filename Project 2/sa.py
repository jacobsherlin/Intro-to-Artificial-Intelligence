import numpy as np
import sys
from SumofGaussians import SumofGaussians

def main():
    #command line argument (seed, number of dimensions, number of gaussians)
    seed_value, dimensions, num_centers = map(int, sys.argv[1:4])

    #random number generator
    random_gen = np.random.default_rng(seed_value)

    #initialize sog function
    sog_function = SumofGaussians(dimensions, num_centers, random_gen)

    #simulated annealing
    max_iter = 100000
    initial_temperature = 1.0
    minimum_temperature = 1e-8
    cooling_rate = 0.995
    epsilon_value = 1e-8  #small tolerance for stopping condition

    #start at a random point in the [0, 10]^d cube
    current_position = random_gen.uniform(0, 10, size=dimensions)
    G_current = sog_function.Evaluate(current_position)

    #simulated annealing loop
    for iteration in range(max_iter):
        log_current_state(current_position, G_current)

        #generate a new point by changing current point
        new_position = perturb_point(current_position, random_gen, dimensions)
        G_new = sog_function.Evaluate(new_position)

        #update temperature
        temperature = max(initial_temperature * (cooling_rate ** iteration), minimum_temperature)

        #metropolis criterion
        if should_accept(G_current, G_new, temperature, random_gen):
            current_position, G_current = new_position, G_new

        #stop if temperature is too low or if improvement is minor
        if temperature < minimum_temperature or abs(G_new - G_current) < epsilon_value:
            break

    log_final_state(current_position, G_current)

def log_current_state(coords, value):
    print(format_coordinates(coords), f"{value:.8f}")

def log_final_state(coords, value):
    log_current_state(coords, value)

def format_coordinates(coords):
    return " ".join(f"{coord:.8f}" for coord in coords)

def perturb_point(position, rng, dims):
    perturbation = rng.uniform(-0.05, 0.05, size=dims)
    new_position = np.clip(position + perturbation, 0, 10)  #make sure its within bounds
    return new_position

def should_accept(G_current, G_new, temperature, rng):
    return G_new > G_current or rng.uniform() < np.exp((G_new - G_current) / temperature)

if __name__ == "__main__":
    main()
