import numpy as np

if __name__ == '__main__':
    n = int(input())
    lst = list(map(int, input().split()))
    print(lst)
    
    arr = np.array(lst)
    print(arr)
    arrList = np.unique(arr)
    arr = np.sort(arrList)[::-1]
    print(arr)
    
    print(arr[1])