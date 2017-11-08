import csv
import numpy as np

def main():
    data = get_data("country.csv")
    #print(len(data[1]))
    #print(sq_distance(a,b))
    centroids = set_centroids(data, 5)
    for i in range(16):
        print("update ", i+1)
        update_centroids(centroids, data, cluster(centroids,data))


def get_data(file):
    """
    Extract the data from the csv file
    :param file:
    :return:
    """
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            try:
                data.append(np.array(row[1:], dtype=float))
            except ValueError:
                pass
        return data

def sq_distance(x,y):
    """
    Calculate the sqaured distance between two vectors
    :param x: first vector
    :param y: second vector
    :return:
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
    centroids=[]
    for i in range(k):
        centroids.append(data[i])
    return centroids

def cluster(centroids, data):
    """
    Generate a list for each data row that tells us which centroid it is closest to
    :param centroids:
    :param data:
    :return:
    """
    clusters = [0 for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(1,len(centroids)):
            # compare distances of current centroid to previous
            if sq_distance(data[i],centroids[j]) < sq_distance(data[i], centroids[clusters[i]]):
                clusters[i] = j
    return clusters

def update_centroids(centroids, data, clusters):
    """
    calculate the average location for each cluster
    :param centroids:
    :param data:
    :param cluster:
    :return:
    """
    cost = 0
    for i in range(len(centroids)):
        tmp_centroid = np.array([0 for _ in range (len(centroids[i]))])
        num_added = 0

        for j in range(len(data)):
            if clusters[j] == i:
                tmp_centroid = np.add(tmp_centroid, data[j])
                num_added += 1
        centroids[i] = tmp_centroid/num_added
        # add to the cost for the new centroids
        for k in range(len(data)):
            if clusters[k] == i:
                cost += sq_distance(data[k], centroids[i])
    print("cost for update: {}\n".format( cost/len(data)))
    return centroids

if __name__ == "__main__":
    main()