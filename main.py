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

def readData(filepath: str) -> List[Task]:
    """
        Read data from a ".txt" file and return an indexed array.
        
        Parameters:
        - `filepath: str` - path to the data file
        
        Returns:
        - `List[Task]` - array of `Task` objects
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

def printData(data: List[Task]) -> None:
    """
        Print array `data` of `Task` elements. 
        
        Created for conveniense.
        Params:
        - `data:  List[Task]` - array of `Task` objects
    """
    print("-----------------")
    for item in data:
        print(item)
    print("-----------------")

def getPenalty(data: List[Task]) -> int:
    """
        Get total penalty for `data` in provided order.
        
        Params:
        - `data: List[Task]` - array of `Task` objects. Penalty is calculated
            for the order given by this array.
        
        Returns:
        - `int` - total penalty value
    """
    data = np.asarray(data.copy())
    if len(data) < 1:
        return 0
    t = 0
    penalty = 0
    for item in data:
        t += item.p
        penalty += getTaskPenalty(item, t)
    return penalty

def getTaskPenalty(task: Task, t: int) -> None:
    """
        Get penalty value for a given task at a given time.
        
        Params:
        - `task: Task` - `Task` object to caltulcate penalty for
        - `t: int` - time to calculate amount of delay
        
        Returns:
        - `int` - penalty value
    """
    return task.w * (t - task.d) if t>task.d else 0

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
    