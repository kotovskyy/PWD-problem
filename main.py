import time
import math
import numpy as np
from typing import List

class Task:
    """
        Class repesenting `Task` object for PWD problem.
    """
    def __init__(self, id: int, p: int, w: int, d: int) -> None:
        """
            - `id: int` - task's id based on the order in the dataset
            - `p: int` - tasks execution time
            - `w: int` - 'weight' or cost per one unit of delay
            - `d: int` - task's completion deadline
        """
        self.id = id
        self.p = p
        self.w = w
        self.d = d
    
    def __repr__(self):
        return f"{self.id} {self.p} {self.w} {self.d}"

def readData(filepath: str):
    """
        Read data from a ".txt" file and return an indexed array.
        
        Parameters:
        - `filepath: str` - path to the data file
        
        Returns:
        - array of `Task` objects
    """
    with open(filepath) as file:
        data = file.readlines()
    
    # remove newlines and split data items
    data = [l.strip().split(" ") for l in data]
    # get number of elements from data - first line of the file
    nItems = int(data[0][0])
    data = np.asarray(data[1:], dtype=np.int64) # convert str to int
    indices = np.arange(0, nItems).reshape(nItems, 1)
    data = np.hstack((indices, data))
    items = []
    for d in data:
        items.append(Task(d[0], d[1], d[2], d[3]))
    
    return items

def getTotalTime(data):
    sum = 0
    for item in data:
        sum += item.p
    return sum

def getPenalty(data):
    data = np.asarray(data.copy())
    C = getTotalTime(data)
    t = 0
    penalty = 0
    for item in data:
        t += item.p
        penalty += item.w * (t - item.d) if (t-item.d)>0 else 0
    return penalty

def printData(data):
    for item in data:
        print(item)

def optimize(data):
    data = np.asarray(data.copy())
    n_subtasks = 2**len(data)
    start = np.zeros((len(data)), dtype=bool)
    printData(data[start])
    print(start)

def main():
    data = readData("data/data0.txt")
    printData(data)
    penalty = getPenalty(data)
    time = getTotalTime(data)
    print(f"Total time: {time}")
    print(f"Penalty: {penalty}")
    optimize(data)
    
if __name__ == "__main__":
    main()
    