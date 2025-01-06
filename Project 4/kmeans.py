'''
Name: Jacob Sherlin
Due Date: 4 December 2024
Course: CSCI-4350-001
Project: OLA #4
Project Description: Develop a program in python that calculates a K-means clustering of a provided set of input training data, assigns classification labels to each cluster using a majority vote, and then reports the classification performance on a separate set of input validation data.

A.I. Disclaimer: All work for this assignment was completed by myself and
entirely without the use of artificial intelligence tools such as ChatGPT, MS
Copilot, other LLMs, etc.

'''

import sys 
import numpy as np  
from collections import Counter  

def load_data(filename):  #load data from file
    data = []  
    with open(filename, 'r') as file:  
        for line in file:  
            values = list(map(float, line.strip().split()))  
            data.append(values)  
    return np.array(data)  

def initialize_centroids(data, k):  #initialize centroids
    return data[:k, :-1]  

def assign_clusters(data, centroids):  #assign data points to clusters
    clusters = []  
    for point in data:  
        distances = np.linalg.norm(point[:-1] - centroids, axis=1)  
        clusters.append(np.argmin(distances))  
    return np.array(clusters)  

def update_centroids(data, clusters, k):  #update centroids
    new_centroids = []  
    for i in range(k):  
        cluster_points = data[clusters == i, :-1]  
        new_centroids.append(np.mean(cluster_points, axis=0))  
    return np.array(new_centroids)  

def majority_vote(labels):  #perform majority vote on labels
    vote = Counter(labels)  
    return vote.most_common(1)[0][0]  

def kmeans_clustering(training_data, k):  #define the K-means clustering function
    centroids = initialize_centroids(training_data, k)  
    clusters = assign_clusters(training_data, centroids)  
    
    while True:  
        new_centroids = update_centroids(training_data, clusters, k)  
        new_clusters = assign_clusters(training_data, new_centroids)  
        if np.array_equal(clusters, new_clusters):  
            break  
        clusters = new_clusters  
        centroids = new_centroids  
    
    labels = []  
    for i in range(k):  
        cluster_labels = training_data[clusters == i, -1]  
        labels.append(majority_vote(cluster_labels))  
    
    return centroids, labels  

def classify(validation_data, centroids, labels):  #classify data points
    correct = 0  
    for point in validation_data:  
        distances = np.linalg.norm(point[:-1] - centroids, axis=1)  
        cluster = np.argmin(distances)  
        if labels[cluster] == point[-1]:  
            correct += 1  
    return correct  

def main():  #main function
    if len(sys.argv) != 4:  
        sys.exit(1)  
    
    k = int(sys.argv[1])  
    training_file = sys.argv[2]  
    validation_file = sys.argv[3]  
    
    training_data = load_data(training_file)  
    validation_data = load_data(validation_file)  
    
    centroids, labels = kmeans_clustering(training_data, k)  
    correct_classifications = classify(validation_data, centroids, labels)  
    
    print(correct_classifications)  

if __name__ == "__main__":  
    main()  
