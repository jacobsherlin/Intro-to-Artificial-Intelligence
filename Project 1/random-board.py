'''
Name: Jacob Sherlin
Due Date: 30 September 2024
Course: CSCI-4350-001
Project: OLA #1
File  Description: Input 8-puzzle from command line and shuffle with randomly generate number a certain amount of times.

A.I. Disclaimer: All work for this assignment was completed by myself and
entirely without the use of artificial intelligence tools such as ChatGPT, MS
Copilot, other LLMs, etc.

'''

import sys
import numpy.random as random

#if variables are not passed on command line
if (len(sys.argv) != 3):
    print()
    print("Correct usage: %s [seed] [number of random moves]" %(sys.argv[0]))
    print()
    sys.exit(1)

def shuffle(puzzle):
    #seed
    rng = random.default_rng(int(sys.argv[1]))
    #cycles for loop
    cycles=int(sys.argv[2])
    for j in range(cycles):
        #range from 1-8
        rand = rng.integers(1,8)
        #find location of 0 and randz
        location_zero = puzzle.index(0)
        location_rand = puzzle.index(rand)
        #swap 0 and rand positions
        puzzle[location_zero], puzzle[location_rand] = puzzle[location_rand], puzzle[location_zero]

def write(puzzle):
    for i in puzzle:
        print(i, end=' ')
    print()  # newline at the end

def main():
    #get puzzle from standard input
    data = sys.stdin.read().split()
    #convert to list of integers
    puzzle = [int(i) for i in data]
    #shuffle and print
    shuffle(puzzle)
    #write to output
    write(puzzle)

main()