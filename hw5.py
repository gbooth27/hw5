import csv
import numpy as np

def main():
    print("running")
    data = get_data("country.csv")
    print(data)
    print(len(data[1]))
    print(sq_distance(data[2],data[4]))


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
                print("row 1")
        return data

def sq_distance(x,y):
    # NEED TO FIX
    dist = 0
    for i in len(x):
        dist += (x[i]*y[i]) * (x[i]*y[i])
    return dist



if __name__ == "__main__":
    main()