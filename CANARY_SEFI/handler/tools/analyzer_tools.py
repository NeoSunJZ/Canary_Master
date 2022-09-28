def calc_average(array):
    count = len(array)
    if count == 0:
        return 0
    else:
        return sum(array) / count