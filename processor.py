import numpy as np

def read_matrix(path, astype=np.float64):
  with open(path, 'r') as f:
    arr = []
    for line in f:
      arr.append([(token if token != '*' else -1) for token in line.strip().split()])
    return np.asarray(arr).astype(astype)

def cart2hom(arr):
  """ Convert catesian to homogenous points 
    by appending a row of 1s
    Input shape: d x n
    Output shape: (d+1) x n """
  if arr.ndim == 1:
    return np.hstack([arr, 1])
  # arr has shape (dimension x num_points)
  return np.vstack([arr, np.ones(arr.shape[1])])

def hom2cart(arr):
  """ Convert homogenous to catesian points
    by dividing each row by the last row 
    Input shape: d x n
    Output shape: (d-1) x n iff d > 1 """
  # arr has shape: dimensions x num_points
  num_rows = len(arr)
  if num_rows == 1 or arr.ndim == 1:
    return arr

  return arr[:num_rows - 1] / arr[num_rows - 1]

