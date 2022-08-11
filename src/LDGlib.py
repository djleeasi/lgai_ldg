import numpy 

def shufflearrays(arrays:list, seed:int):
    length = None
    length = len(arrays[0])
    if type(length) != int:
        raise Exception('Cannot read array length')
    for array in arrays:
        if length != len(array):
            raise Exception('recieved arrays are not in same length')
    RandomGenerator = numpy.random.RandomState(seed=seed)
    order = numpy.arange(length)
    RandomGenerator.shuffle(order)
    del RandomGenerator
    outputlist = []
    for array in arrays:
        shuffled = array[order]
        outputlist.append(shuffled)
    return outputlist