import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import math


def main(skip):
    data, country_dict = get_data("country.csv")
    ########################################################
    # Include Command line arg -s to skip the graphing step
    ########################################################
    if skip:
        generate_cost_graph(k_max=51, data=data)

    best = [math.inf,[]]
    # run the cluster update for 1 trial
    k = 22
    for _ in range(100):
        centroids = set_centroids(data, k)
        cost = 0
        iters = 0
        converged = False
        # run until converged OR until reach maxiters
        while (not converged):
            prev = cost
            clust = cluster(centroids, data)
            # Update the centroids
            centroids, cost = update_centroids(centroids, data, clust)
            iters += 1
            if cost == prev or iters > 500:
                converged = True
        # update the best cost
        if best[0] > cost:
            best = [cost, clust]

    clusters = best[1]
    # extract which cluster each country is in for printing
    country_cluster_dict = {}
    # get each country's cluster
    for country in range(len(clusters)):
        country_cluster_dict[country] = clusters[country]
    cluster_dict = {}
    # get the countries in each cluster
    for country in country_cluster_dict:
        cluster_number =country_cluster_dict[country]
        if cluster_number in cluster_dict:
            cluster_dict[cluster_number].append(country_dict[country])
        else:
            cluster_dict[cluster_number] = [country_dict[country]]
    # print each cluster and its countries
    for i in range(len(cluster_dict)):
        print("Cluster {}, contains: ".format(i))
        for country in cluster_dict[i]:
            print("    * " + country)
        print()



def generate_cost_graph(k_max, data):
    """
    generates the plot of costs vs k
    :return:
    """
    best_list = []
    # for k 1 -> k_max run K means
    for k in range(1, k_max):
        costs = {}
        # run the cluster update for 1 trial
        for _ in range(10):
            centroids = set_centroids(data, k)
            cost = 0
            iters = 0
            converged = False
            # run until converged OR until reach maxiters
            while (not converged):
                prev = cost
                clust = cluster(centroids, data)
                centroids, cost = update_centroids(centroids, data, clust)
                # print("cost for update: {}".format(cost))
                iters += 1
                if cost == prev or iters > 500:
                    converged = True
            # add the costs to the dict
            costs[cost] = clust
        best = min(costs.keys())
        best_list.append(best)

    plt.plot([i for i in range(1, k_max)], best_list)
    plt.show()


def get_data(file):
    """
    Extract the data from the csv file
    :param file:
    :return:
    """
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        countrys = {}
        i = 0
        for row in reader:
            try:

                data.append(np.array(row[1:], dtype=float))
                countrys[i] = row[0]
                i+=1
            except ValueError:
                pass
        return data, countrys

def sq_distance(x,y):
    """
    Calculate the squared distance between two vectors
    :param x: first vector
    :param y: second vector
    :return: squared distance
    """
    dist = np.linalg.norm(x-y)**2
    return dist

def set_centroids(data,k):
    """
    Sets the centroids as the first three vectors for debugging
    :param data:
    :param k:
    :return:
    """
    return random.sample(data, k)

def cluster(centroids, data):
    """
    Generate a list for each data row that tells us which centroid it is closest to
    :param centroids: list of current centroids
    :param data: data to cluster the centroids
    :return: list of which centroid each data point belongs to.
    """
    clusters = [0 for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(centroids)):
            # compare distances of current centroid to previous
            # if it is smaller update it.
            if sq_distance(data[i],centroids[j]) < sq_distance(data[i], centroids[clusters[i]]):
                clusters[i] = j
    return clusters

def update_centroids(centroids, data, clusters):
    """
     Update the centroids and calculate the average location for each cluster
    :param centroids: list of centroids to update
    :param data: data for clustering
    :param cluster: which data point goes with which cluster
    :return: updated centroids
    """
    cost = 0
    for i in range(len(centroids)):
        tmp_centroid = np.array([0 for _ in range (len(centroids[i]))])
        num_added = 0
        #
        for j in range(len(data)):
            if clusters[j] == i:
                tmp_centroid = np.add(tmp_centroid, data[j])
                num_added += 1
        centroids[i] = tmp_centroid/num_added
        # add to the cost for the new centroids
        for k in range(len(data)):
            if clusters[k] == i:
                cost += sq_distance(data[k], centroids[i])
    cost = cost / len(data)


    return centroids, cost

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', "-s", dest='feature', action='store_false', help="use to skip creation of word files")
    args = parser.parse_args()
    main(args.feature)