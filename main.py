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

def getTotalTime(data: List[Task], orderID: int) -> int:
    """
        Get total time needed to complete set of tasks in the order, specified 
        by the `orderID`. Ex.:  
        
        `orderID = 3 = (0011)_2`; `i` in `[0, len(data))`  
        
        `1 << i` produces values `0001`, `0010`, `0100` etc. Next, bitwise AND 
        operation is used to see if task with id `1 << i` is in the order
        
        Params:
        - `data: List[Task]` - 
        - `orderID: int` - id of the tasks order `[0, 2**len(data))`
        
        Returns:
        - `int` - total completion time
    """
    sum = 0
    for i in range(len(data)):
        if (1 << i) & orderID:
            sum += data[i].p
    return sum

def PD_Algorithm(data: List[Task]) -> List[int]:
    """
        PD algorithm for PWD problem solving.
        `N` is the number of tasks in `data` array. Number of possible tasks
        combinations is `2**N`. Array `table` contains smallest possible penalty
        value for a given task combination. Indicies in `table` are in range 
        `[0, 2**N]` (index `0` - no tasks are taken, index `2**N-1 - all tasks).
        Algorithm goes through all possible combinations finding optimal order, 
        which is then being saved to `table_order`, where every element with
        indicies `[0, 2**N]` contains `List[int]`. This array corresponds to
        task's ids in optimal order. 
        
        Params:
        - `data: List[Task]` - array of tasks to optimize
        
        Returns:
        - `List[int]` - array of task's ids in optimized order
    """
    N = len(data)
    table = [0] + [math.inf for _ in range(2**N-1)]
    table_order = [[] for _ in range(2**N)]

    for i in range(1, 2**N):
        time = getTotalTime(data, i)
        k, j, index = 0, 1, 0
        while (j <= i):
            if i & j:
                prev = table[i-j]
                penalty = getTaskPenalty(data[k], time)
                if not math.isinf(prev):
                    penalty += prev
                
                if table[i] > penalty:
                    table[i] = penalty
                    table_order[i] = table_order[i-j][:]
                    index = k
                elif (table[i] == penalty) and (table_order[i] > table_order[i-j]):
                    table_order[i] = table_order[i-j][:]
                    index = k
            j <<= 1
            k += 1
            
        table_order[i].append(data[index].id)
        
    return table_order[-1]


def calculate_time(func):
    """
        Decorator to calculate total execution time of a function.
    """
    def inner(*args, **kwargs):
        import time
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        totalTime = end - start
        print(f"Execution time: {totalTime:.3} s")
        
    return inner

@calculate_time
def testSolution(filename: str) -> None:
    """
        Solve PWD problem for a dataset with `filename` name.
        
        Params:
        - `filename: str` - name of the data file. Must be of type ".txt" and 
        placed into `data/` folder. File must have structure:
            - 1st line: number of tasks in the dataset
            - N line: task described with 3 `int`s separated by spaces: "p w d"
    """
    filepath = f"data/{filename}"
    data = readData(filepath)
    print(f"DATASET : {filename}")
    # printData(data)
    order = PD_Algorithm(data)
    totalPenalty = getPenalty(np.array(data)[order])
    
    print(f"Total penalty: {totalPenalty}")
    print(f"Order: {" ".join([str(item + 1) for item in order])}")

def testMultiple(filenames):
    for filename in filenames:
        testSolution(filename)

def main():
    filenames = ["data0.txt", "data1.txt", "data2.txt", "data3.txt",
                 "data4.txt", "data5.txt", "data6.txt", "data7.txt",
                 "data8.txt", "data9.txt", "data10.txt"]
    testMultiple(filenames)
        
if __name__ == "__main__":
    main()