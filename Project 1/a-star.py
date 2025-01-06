'''
Name: Jacob Sherlin
Due Date: 30 September 2024
Course: CSCI-4350-001
Project: OLA #1
File Description: Use A* search algorithm to solve 8-puzzle problem

A.I. Disclaimer: All work for this assignment was completed by myself and
entirely without the use of artificial intelligence tools such as ChatGPT, MS
Copilot, other LLMs, etc.

'''

import sys #used for command line input
from copy import deepcopy #used to maintain puzzle matrix

#goal state
GOAL = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#moves that can be made for each direction
MOVES = {"up": [-1, 0], "down": [1, 0], "left": [0, 1], "right": [0, -1]} 

def read_puzzle():
    #receive puzzle from standard input
    line = input()
    #split and convert to integers
    numbers = list(map(int, line.split())) 
    #reshape into a 3x3 matrix
    matrix = [numbers[i:i+3] for i in range(0, len(numbers), 3)]
    
    return matrix

class Node:
    id_counter = 0  #class variable to keep track of IDs
    def __init__(self, current_node, previous_node, g, h, direction):
        self.id = Node.id_counter  #assign unique ID
        Node.id_counter += 1  #increment ID counter
        self.current_node = current_node
        self.previous_node = previous_node
        self.g = g
        self.h = h
        self.direction = direction

    def f(self):
        #calculates total cost function
        return self.g + self.h

def get_pos(current_state, element):
    #finds position of specific element in matrix
    for row in range(len(current_state)):
        if element in current_state[row]:
            return row, current_state[row].index(element)

def get_adjacent_nodes(node):
    #generate adjacent nodes from the current node state
    adjacent_nodes = []
    empty_pos = get_pos(node.current_node, 0)
    #loop through possible moves
    for direction, move in MOVES.items():
        new_pos = (empty_pos[0] + move[0], empty_pos[1] + move[1])        
        #check if new position is within bounds
        if 0 <= new_pos[0] < len(node.current_node) and 0 <= new_pos[1] < len(node.current_node[0]):
            #create a new state by swapping empty position with the adjacent position
            new_state = deepcopy(node.current_node)
            new_state[empty_pos[0]][empty_pos[1]], new_state[new_pos[0]][new_pos[1]] = new_state[new_pos[0]][new_pos[1]], 0
            #create new node and add it to the list of adjacent nodes
            adjacent_nodes.append(Node(new_state, node.current_node, node.g + 1, calc_cost(new_state), direction))

    return adjacent_nodes

def calc_cost(current_state):
    #takes in heuristic choice to calculate cost (f)
    if len(sys.argv) > 1:
        choice = int(sys.argv[1])
    else:
        #if choice is not 1, 2, or 3, default to 0
        choice = 0

    cost = 0
    if choice == 1:
        #displaced tiles heuristic
        for row in range(len(current_state)):
            for col in range(len(current_state[0])):
                if current_state[row][col] != GOAL[row][col] and current_state[row][col] != 0:
                    cost += 1
    elif choice == 2:
        #manhattan distance heuristic
        for row in range(len(current_state)):
            for col in range(len(current_state[0])):
                if current_state[row][col] != 0:
                    target_pos = get_pos(GOAL, current_state[row][col])
                    cost += abs(row - target_pos[0]) + abs(col - target_pos[1])
    elif choice == 3:
        #nine multiplied by number of displaced tiles
        for row in range(len(current_state)):
            for col in range(len(current_state[0])):
                if current_state[row][col] != GOAL[row][col] and current_state[row][col] != 0:
                    cost += 9
    return cost

def get_best_node(frontier):
    #finds the node with lowest total cost (f) in the open set
    best_node = None
    best_f = float('inf')  #initialize with infinity

    for node in frontier.values():
        if node.f() < best_f or (node.f() == best_f and node.id > best_node.id):
            best_node = node
            best_f = node.f()
    return best_node

def build(closed_list):
   #generate path from initial state to solution
    node = closed_list[str(GOAL)]
    path = []
    depth = 0
    #trace back from the goal node to initial state
    while node.direction:
        path.append({'node': node.current_node, 'direction': node.direction})
        node = closed_list[str(node.previous_node)]
        depth += 1
    #append initial node to path
    path.append({'node': node.current_node, 'direction': ''})
    #reverse path to get it from initial state to solution
    path = path[::-1]
    
    return path, depth

def print_puzzle(array):
    #print puzzle
    for row in array:
        print(" ".join(str(element) for element in row))


def main(puzzle):
    #frontier contains open list of nodes
    frontier = {str(puzzle): Node(puzzle, puzzle, 0, calc_cost(puzzle), "")}
    closed_list = {}  #closed list for visited nodes, even tho it's a set
    #stats
    V = 0  #total number of nodes visited/expanded
    max_nodes_in_memory = 1  #starting node (1)

    while frontier:
        current_node = get_best_node(frontier)
        closed_list[str(current_node.current_node)] = current_node
        V += 1  #increment the number of visited/expanded nodes
        #check if the solution is found
        if current_node.current_node == GOAL:
            path, depth = build(closed_list)
            max_nodes_in_memory = max(max_nodes_in_memory, len(closed_list) + len(frontier))
            return path, depth, V, max_nodes_in_memory
        #generate and process adjacent nodes
        adjacent_nodes = get_adjacent_nodes(current_node)
        for node in adjacent_nodes:
            if (str(node.current_node) in closed_list) or \
               (str(node.current_node) in frontier and frontier[str(node.current_node)].f() < node.f()):
                continue
            frontier[str(node.current_node)] = node
        del frontier[str(current_node.current_node)]  #delete processed node
        #track max number of nodes stored in memory
        max_nodes_in_memory = max(max_nodes_in_memory, len(closed_list) + len(frontier))

    return None, None, V, max_nodes_in_memory

if __name__ == '__main__':
    #read in puzzle and convert to 3x3 matrix
    puzzle = read_puzzle()
    path, depth, V, N = main(puzzle)
    #print stats
    print('V=',V)
    print('N=',N)
    print('d=',depth)
    b = 0
    if depth > 0:
        b = N ** (1 / depth)
    print('b=',b)
    print()
    #print path for solution
    for step in path:
        print_puzzle(step['node'])
        print()
