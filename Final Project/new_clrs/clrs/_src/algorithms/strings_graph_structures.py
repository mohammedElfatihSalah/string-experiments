import numpy as np

def get_predecessor(T,P):
  # copy the inputs
  T = np.copy(T)
  P = np.copy(P)

  P_size = P.shape[0]
  T_size = T.shape[0]

  adj = np.zeros((P_size + T_size,P_size + T_size))

  # predecessor for Text
  for i in range(1,T_size):
    adj[i, i-1] = 1
  
  # predecessor for Pattern
  for i in range(1,P_size):
    adj[T_size+i, T_size+i-1] = 1
  
  return adj

def get_graph_struct(T, P, h_i, h_j, h_s):
  # copy the inputs
  T = np.copy(T)
  P = np.copy(P)

  P_size = P.shape[0]
  T_size = T.shape[0]

  adj = np.zeros((P_size + T_size,P_size + T_size))  

  for i in range(h_s+1, h_i):
    adj[i, h_i] = 1

  adj[T_size, T_size + h_j] = 1
  
  for i in range(T_size):
    adj[i, T_size+h_j] = 1
  
  for i in range(P_size):
    adj[i+T_size, h_i] = 1
  
  return adj
  


def get_seq_mat(T,P):
  n  = T.shape[0]
  m  = P.shape[0] 

  mat = np.eye((n+m))
  # connect each character to its previous
  for i in range(1,n+m):
    if i == n:
      # don't do it for the start of the pattern
      continue
    mat[i, i-1] = 1

  # connect each character in text to its equal charcter in the pattern
  for i in range(n):
    for j in range(m):
      if T[i] == P[j]:
        mat[i, j+n] = 1
        mat[j+n, i] = 1
  
  # connect the start of the pattern with all character upfront
  mat[n, n+1:] = 1
  return mat

def get_t(T, P, s):
  i = s
  j = 0

  N = T.shape[0]
  M = P.shape[0]
  while i < N:
    if T[i] != P[j]:
      return i
    j +=1
    i +=1
    if j >= M:
      return i
  return N - 1
  


def get_bipartite_mat(T, P, s, num_classes=3):
  '''
  args
  -----------------------------
    T: the text
    P: the pattern
    s: current hint s
  
  returns
  -----------------------------
  mat: constructed mat as the following:
       1- all irrelevant edges will have a value of 0
       2- relevant edges will have a value of 1 if they are equal, 
          otherwise they will have a value of 2
  '''
  # length of the text
  N = T.shape[0]
  # length of the pattern
  M = P.shape[0]

  mat = np.zeros((N+M, N+M), dtype=np.int)
  
  t = get_t(T, P, s)
  for i in range(M):
    p_char = P[i]
    for j in range(s,t):
      t_char = T[j]
      if t_char == p_char:
        mat[j, i+N] = 1
        mat[i+N, j] = 1
      else:
        mat[j, i+N] = 2
        mat[i+N, j] = 2
  
  one_hot_mat = np.zeros((N+M, N+M, num_classes), dtype=np.int)
  for i in range(len(mat)):
    for j in range(len(mat[0])):
      class_id = mat[i, j]
      one_hot_mat[i, j, class_id] = 1
  
  return one_hot_mat


#=== *** ===#
def get_everything_matched_to_this_point(T, P, s):
  '''
  return a binary mask for the pattern
  '''

  result = np.zeros(T.shape[0] + P.shape[0],dtype=np.int)
  i = s
  j = 0
  while j < P.shape[0]:
    if T[i] == P[j]:
      result[T.shape[0]+j] = 1
      i+=1
      j+=1
    else:
      break
  return result

def get_bipartite_mat_from_pattern_to_text(T, P, s):
  # length of the text
  N = T.shape[0]
  # length of the pattern
  M = P.shape[0]

  mat = np.zeros((N+M, N+M), dtype=np.int)
  
  for i in range(M):
    p_char = P[i]
    for j in range(s,N):
      t_char = T[j]
      if t_char == p_char:
        mat[j, i+N] = 1
        mat[i+N, j] = 1
      else:
        mat[j, i+N] = 2
        mat[i+N, j] = 2


def get_seq_mat_i_j(T, P , i ,j, s):
  n  = T.shape[0]
  m  = P.shape[0] 

  mat = np.zeros((n+m, n+m))
  # connect each character to its previous
  # for i in range(1,n+m):
  #   if i == n:
  #     # don't do it for the start of the pattern
  #     continue
  #   mat[i, i-1] = 1

  # connect node i with node j
  mat[i, j+n] = 1
  mat[j+n, i] = 1

  # connect node s with i
  mat[s, i] = 1
  mat[i,s] = 1

  # connect first node in P with node 
  mat[n,n+j] = 1
  
  return mat

def get_edge_mat(T, P, start, end):
  '''
  edge between start and end
  '''
  mat = np.zeros((n+m,n+m))
  mat[start, end] = 1
  return mat
