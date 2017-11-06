import csv
import numpy as np

def get_data(file):
    """
    Extract the data from the csv file
    :param file:
    :return:
    """
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)

if __name__ == '__main__':
    get_data("country.csv")