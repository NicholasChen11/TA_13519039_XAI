def Euclidean(A, B, *_):
  n_features = len(A)
  totalSum = 0
  
  for i in range(n_features):
    totalSum += (A[i] - B[i])**2
    
  return totalSum**(1/2)

def Manhattan(A, B, *_):
  n_features = len(A)
  totalSum = 0
  
  for i in range(n_features):
    totalSum += abs(A[i] - B[i])
    
  return totalSum

def Chebyshev(A, B, *_):
  n_features = len(A)
  maxDistance = 0
  
  for i in range(n_features):
    distance = abs(A[i] - B[i])
    
    if distance > maxDistance:
      maxDistance = distance
    
  return maxDistance
  
def Minkowski(A, B, p, *_):
  n_features = len(A)
  totalSum = 0
  
  for i in range(n_features):
    totalSum += (abs(A[i] - B[i]))**p
    
  return totalSum**(1/p)
