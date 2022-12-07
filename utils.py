def most_frequent(List):
  return max(set(List), key = List.count)

def countDifference(A, B):
  difference = [0 for _ in range(len(A))]
  
  for i in range(len(A)):
    if A[i] == B[i]:
      difference[i] = 0
    else: 
      difference[i] = 1
  
  return difference.count(1)

def weightedSum(array):
  totalSum = 0
  for i, k in enumerate(array):
    totalSum = 1/(i+1) * k
    
  return totalSum

def arrayAddition(array_1, array_2):
  array_result = array_1.copy()
  for i in range(len(array_2)):
    array_result[i] += array_2[i]
  
  return array_result

def arrayMultConst(array, c):
  array_result = array.copy()
  for i in range(len(array)):
    array_result[i] *= c
  
  return array_result