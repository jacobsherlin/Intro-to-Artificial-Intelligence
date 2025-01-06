'''
Name: Jacob Sherlin
Due Date: 15 November 2024
Course: CSCI-4350-001
Project: OLA #3
Project Description: Develop a software agent in Python to learn an ID3 decision tree from labeled classification data.

A.I. Disclaimer: All work for this assignment was completed by myself and
entirely without the use of artificial intelligence tools such as ChatGPT, MS
Copilot, other LLMs, etc.

'''

import sys
import numpy as np
from collections import Counter
import math

class TreeNode:
    #node structure for decision tree
    def __init__(self, feature_index=None, split_value=None, left_node=None, right_node=None, label=None):
        #feature_index for splitting, split_value for threshold
        #left_node and right_node are child nodes, label for leaf nodes
        self.feature_index = feature_index
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        self.label = label

#calculate entropy based on class distribution
def compute_entropy(class_values):
    value_counts = Counter(class_values)
    probabilities = [count / len(class_values) for count in value_counts.values()]
    #entropy formula - sum of -p*log2(p) for all probabilities
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

#calculate information gain for a given feature and split value
def calculate_information_gain(data, feature_index, split_value):
    #split data into left and right subsets based on split_value
    left_data = data[data[:, feature_index] < split_value]
    right_data = data[data[:, feature_index] >= split_value]
    
    #obtain class values for full data and subsets
    all_class_values = data[:, -1]
    left_class_values = left_data[:, -1]
    right_class_values = right_data[:, -1]
    
    #calculate entropy for full data and subsets
    total_entropy = compute_entropy(all_class_values)
    left_entropy = compute_entropy(left_class_values) if len(left_class_values) > 0 else 0
    right_entropy = compute_entropy(right_class_values) if len(right_class_values) > 0 else 0
    
    #calculate weighted entropy based on split proportions
    weight_left = len(left_data) / len(data)
    weight_right = len(right_data) / len(data)
    
    #calculate information gain from this split
    gain = total_entropy - (weight_left * left_entropy + weight_right * right_entropy)
    return gain

#identify the best feature and split value for splitting the data
def identify_best_split(data):
    max_gain = -1  #initialize to a low value for comparison
    optimal_feature = None
    optimal_split_value = None

    #iterate over each feature in data except class labels
    for feature_index in range(data.shape[1] - 1):
        sorted_indices = np.argsort(data[:, feature_index])
        sorted_data = data[sorted_indices]
        
        #iterate through sorted data to find unique split values
        for i in range(1, len(data)):
            if sorted_data[i - 1, feature_index] != sorted_data[i, feature_index]:
                split_value = (sorted_data[i - 1, feature_index] + sorted_data[i, feature_index]) / 2
                gain = calculate_information_gain(data, feature_index, split_value)
                
                #check for max gain or prioritize smaller feature_index if gain ties
                if gain > max_gain or (gain == max_gain and (optimal_feature is None or feature_index < optimal_feature)):
                    max_gain = gain
                    optimal_feature = feature_index
                    optimal_split_value = split_value

    return optimal_feature, optimal_split_value

#find the majority label in a set of class values, with tie-breaking
def find_majority_label(class_values):
    value_counts = Counter(class_values)
    common_values = value_counts.most_common()
    #break ties by choosing smallest label if necessary
    if len(common_values) > 1 and common_values[0][1] == common_values[1][1]:
        return min(common_values[0][0], common_values[1][0])
    return common_values[0][0]

#recursively create the decision tree
def create_decision_tree(data):
    class_values = data[:, -1]
    #if only one unique class value, create a leaf node
    if len(set(class_values)) == 1:
        return TreeNode(label=class_values[0])

    feature_index, split_value = identify_best_split(data)
    #if no feature found, return leaf with majority label
    if feature_index is None:
        return TreeNode(label=find_majority_label(class_values))

    #split data and recursively create left and right nodes
    left_data = data[data[:, feature_index] < split_value]
    right_data = data[data[:, feature_index] >= split_value]
    left_node = create_decision_tree(left_data) if len(left_data) > 0 else TreeNode(label=find_majority_label(class_values))
    right_node = create_decision_tree(right_data) if len(right_data) > 0 else TreeNode(label=find_majority_label(class_values))

    return TreeNode(feature_index=feature_index, split_value=split_value, left_node=left_node, right_node=right_node)

#classify an example by traversing the decision tree
def make_prediction(tree, example):
    #if label is set, it's a leaf node, so return label
    if tree.label is not None:
        return tree.label
    #decide path based on feature threshold comparison
    if example[tree.feature_index] < tree.split_value:
        return make_prediction(tree.left_node, example)
    else:
        return make_prediction(tree.right_node, example)

#load data from file
def load_data(filename):
    data = np.loadtxt(filename)
    #ensure data is 2-dimensional
    if len(data.shape) < 2:
        data = np.array([data])
    return data

#main function to train and validate the decision tree
def main():
    training_file = sys.argv[1]
    validation_file = sys.argv[2]
    #load training and validation data
    training_data = load_data(training_file)
    validation_data = load_data(validation_file)

    #build tree from training data
    tree = create_decision_tree(training_data)
    #count correctly classified examples
    correct_predictions = sum(1 for example in validation_data if make_prediction(tree, example) == example[-1])

    print(correct_predictions)

if __name__ == "__main__":
    main()
