import numpy as np

def find_min(array, number_of_mins):
    min_ix = np.zeros(number_of_mins) - 1
    items = np.zeros(number_of_mins)
    for i in range(len(array)):
        item = array[i]
        if i == 0:
            items[0] = item
            min_ix[0] = 0
            continue
        maxj = min(i - 1, number_of_mins - 1)
        for j in range(maxj, -1, -1):
            if item < items[j] or min_ix[j] == -1:
                if j < number_of_mins - 1:
                    min_ix[j + 1] = min_ix[j]
                    items[j + 1] = items[j]
                    if j == 0:
                        min_ix[0] = i
                        items[0] = item
            else:
                if j < number_of_mins - 1:
                    min_ix[j + 1] = i
                    items[j + 1] = item
                break
    return min_ix