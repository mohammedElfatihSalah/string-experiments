# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Strings algorithm generators.

Currently implements the following:
- Naive string matching
- Knuth-Morris-Pratt string matching (Knuth et al., 1977)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name


from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np
from clrs._src.algorithms.strings_graph_structures import get_seq_mat, get_seq_mat_i_j,get_graph_struct,get_predecessor


_Array = np.ndarray
_Out = Tuple[int, probing.ProbesDict]

_ALPHABET_SIZE = 4


def get_from_i_to_j(T,P,i,j):
  N = T.shape[0]
  M = P.shape[0]

  mat = np.zeros((N+M, N+M), dtype=np.int)

  mat[i, j+N] = 1
  return mat


# def naive_string_matcher(T: _Array, P: _Array) -> _Out:
#   """Naive string matching."""

#   chex.assert_rank([T, P], 1)
#   probes = probing.initialize(specs.SPECS['naive_string_matcher'])

#   T_pos = np.arange(T.shape[0])
#   P_pos = np.arange(P.shape[0])

#   predecessor = get_predecessor(T,P)

#   probing.push(
#       probes,
#       specs.Stage.INPUT,
#       next_probe={
#           'string':
#               probing.strings_id(T_pos, P_pos),
#           'pos':
#               probing.strings_pos(T_pos, P_pos),
#           'key':
#               probing.array_cat(
#                   np.concatenate([np.copy(T), np.copy(P)]), _ALPHABET_SIZE),
#           'pattern_start':probing.mask_one(T.shape[0], T.shape[0] + P.shape[0]),
#           'predecessor':predecessor,
#       })

#   s = 0
#   while s <= T.shape[0] - P.shape[0]:
#     i = s
#     j = 0

#     # mat_i_j = get_seq_mat_i_j(np.copy(T), np.copy(P), i, j, s)
#     # from_i_to_j = get_from_i_to_j(T,P, i, j)

#     adj = get_graph_struct(T,P,i,j,s)

#     probing.push(
#         probes,
#         specs.Stage.HINT,
#         next_probe={
#             'pred_h': probing.strings_pred(T_pos, P_pos),
#             's': probing.mask_one(s, T.shape[0] + P.shape[0]),
#             'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
#             'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
#             'adj': adj,
#             'pred_h': probing.strings_pred(T_pos, P_pos),
#         })

#     while True:
#       if T[i] != P[j]:
#         break
#       elif j == P.shape[0] - 1:
#         probing.push(
#             probes,
#             specs.Stage.OUTPUT,
#             next_probe={'match': probing.mask_one(s, T.shape[0] + P.shape[0])
#             })
#         probing.finalize(probes)
#         return s, probes
#       else:
#         i += 1
#         j += 1

#         adj = get_graph_struct(T,P,i,j,s)

#         # from_i_to_j = get_from_i_to_j(T,P, i, j)
#         # mat_i_j = get_seq_mat_i_j(np.copy(T), np.copy(P), i, j, s)
#         probing.push(
#             probes,
#             specs.Stage.HINT,
#             next_probe={
#                 'pred_h': probing.strings_pred(T_pos, P_pos),
#                 's': probing.mask_one(s, T.shape[0] + P.shape[0]),
#                 'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
#                 'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
#                 'adj': adj,
#                 'pred_h': probing.strings_pred(T_pos, P_pos),    
#             })

#     s += 1

#   # By convention, set probe to head of needle if no match is found
#   probing.push(
#       probes,
#       specs.Stage.OUTPUT,
#       next_probe={
#           'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
#       })
#   return T.shape[0], probes

def get_i_j_mat(T,P, i,j):
  T = np.copy(T)
  P = np.copy(P)

  T_len = T.shape[0]
  P_len = P.shape[0]

  mat = np.zeros((T_len + P_len, T_len + P_len))

  mat[i, j + T_len] = 1
  mat[j + T_len, i] = 1
  return mat

# hope is here

# def naive_string_matcher(T: _Array, P: _Array) -> _Out:
#   """Naive string matching."""
#   #print(f"T {T}")
#   #print(f"P {P}")
#   chex.assert_rank([T, P], 1)
#   probes = probing.initialize(specs.SPECS['naive_string_matcher'])

#   T_pos = np.arange(T.shape[0])
#   P_pos = np.arange(P.shape[0])

#   adj = np.full((T.shape[0] + P.shape[0] , T.shape[0] + P.shape[0]), 1)

#   # debug
#   debug_advance = []

#   probing.push(
#       probes,
#       specs.Stage.INPUT,
#       next_probe={
#           'string':
#               probing.strings_id(T_pos, P_pos),
#           'pos':
#               probing.strings_pos(T_pos, P_pos),
#           'key':
#               probing.array_cat(
#                   np.concatenate([np.copy(T), np.copy(P)]), _ALPHABET_SIZE),
#           'adj': adj
#       })

#   s = 0
#   while s <= T.shape[0] - P.shape[0]:
#     i = s
#     j = 0
#     # debug
#     #print("check: ", T[i], T[j])
#     debug_advance.append(int(T[i] != P[j]))
#     # if s == 0:
#     #   if T[i] == P[j]:
#     #     debug_advance.append(0)
#     #   else:
#     #     debug_advance.append(1)
#     # else:
#     #   debug_advance.append(1)

#     i_j_mat = get_i_j_mat(T,P,i,j)

#     adj2 = np.zeros((T.shape[0] + P.shape[0] , T.shape[0] + P.shape[0]))
#     adj2[i,T.shape[0]:] = 1
#     adj2[T.shape[0]:, i] = 1
#     adj2[:T.shape[0], j + T.shape[0]] = 1
#     adj2[j + T.shape[0], :T.shape[0]] = 1


#     # if s == 0:
#     #   probing.push(
#     #       probes,
#     #       specs.Stage.HINT,
#     #       next_probe={
#     #           'pred_h': probing.strings_pred(T_pos, P_pos),
#     #           's': probing.mask_one(s, T.shape[0] + P.shape[0]),
#     #           'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
#     #           'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
#     #           'advance':1,
#     #           'i_g':0,
#     #           'j_g':0,
#     #           's_g':0,
#     #           'i_j_mat': i_j_mat,
#     #           'adj2':adj2,
#     #       })
#     # else:
#     #   probing.push(
#     #       probes,
#     #       specs.Stage.HINT,
#     #       next_probe={
#     #           'pred_h': probing.strings_pred(T_pos, P_pos),
#     #           's': probing.mask_one(s, T.shape[0] + P.shape[0]),
#     #           'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
#     #           'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
#     #           'advance':1,
#     #           'i_g':0,
#     #           'j_g':0,
#     #           's_g':1,
#     #           'i_j_mat': i_j_mat,
#     #           'adj2':adj2
#     #       })
#     probing.push(
#         probes,
#         specs.Stage.HINT,
#         next_probe={
#             'pred_h': probing.strings_pred(T_pos, P_pos),
#             's': probing.mask_one(s, T.shape[0] + P.shape[0]),
#             'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
#             'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
#             'advance': int(T[i] != P[j]),
#             'i_g':0,
#             'j_g':0,
#             's_g':1,
#             'i_j_mat': i_j_mat,
#             'adj2':adj2
#         })


#     while True:
#       if T[i] != P[j]:
#         break
#       elif j == P.shape[0] - 1:
#         probing.push(
#             probes,
#             specs.Stage.OUTPUT,
#             next_probe={'match': 0,
#             #probing.mask_one(s, T.shape[0] + P.shape[0])
#             })
#         probing.finalize(probes)
#         #debug
#         #print("debug advance >> ", debug_advance)
#         return s, probes
#       else:
#         i += 1
#         j += 1
        
#         i_j_mat = get_i_j_mat(T,P,i,j)

#         adj2 = np.zeros((T.shape[0] + P.shape[0] , T.shape[0] + P.shape[0]))
#         adj2[i,T.shape[0]:] = 1
#         adj2[T.shape[0]:, i] = 1
#         adj2[:T.shape[0], j + T.shape[0]] = 1
#         adj2[j + T.shape[0], :T.shape[0]] = 1
#         #debug
#         debug_advance.append(int(T[i] != P[j]))

#         probing.push(
#             probes,
#             specs.Stage.HINT,
#             next_probe={
#                 'pred_h': probing.strings_pred(T_pos, P_pos),
#                 's': probing.mask_one(s, T.shape[0] + P.shape[0]),
#                 'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
#                 'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
#                'advance':int(T[i] != P[j]),
#                 'i_g':1,
#                 'j_g':1,
#                 's_g':0,
#                 'i_j_mat': i_j_mat,
#                 'adj2':adj2
#             })

#     s += 1

#   # By convention, set probe to head of needle if no match is found
#   probing.push(
#       probes,
#       specs.Stage.OUTPUT,
#       next_probe={
#           'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
#       })
#   #debug
#   #print("debug advance >> ", debug_advance)
#   return T.shape[0], probes

def get_predecessor(T, P):
  nb_text = T.shape[0]   
  nb_pattern = P.shape[0]   
  predecessor = np.eye(nb_pattern + nb_text)

  for i in range(1,nb_text):
    predecessor[i-1,i] = 1
    #predecessor[i,i-1] = 1
  
  for j in range(1 + nb_text, nb_pattern + nb_text):
    predecessor[j-1, j] = 1
    #predecessor[j, j-1] = 1
  
  return predecessor


def naive_string_matcher(T: _Array, P: _Array) -> _Out:
  """Naive string matching."""

  chex.assert_rank([T, P], 1)
  probes = probing.initialize(specs.SPECS['naive_string_matcher'])

  T_pos = np.arange(T.shape[0])
  P_pos = np.arange(P.shape[0])

  adj = np.full((T.shape[0] + P.shape[0] , T.shape[0] + P.shape[0]), 1)

  predecessor = get_predecessor(T, P)

  # debug
  debug_advance = []

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
        'predecessor':predecessor,
          'string':
              probing.strings_id(T_pos, P_pos),
          'pos':
              probing.strings_pos(T_pos, P_pos),
          'key':
              probing.array_cat(
                  np.concatenate([np.copy(T), np.copy(P)]), _ALPHABET_SIZE),
          'adj': adj
      })

  s = 0
  while s <= T.shape[0] - P.shape[0]:
    i = s
    j = 0
    # debug
    #print("check: ", T[i], T[j])
    debug_advance.append(int(T[i] != P[j]))
    # if s == 0:
    #   if T[i] == P[j]:
    #     debug_advance.append(0)
    #   else:
    #     debug_advance.append(1)
    # else:
    #   debug_advance.append(1)

    i_j_mat = get_i_j_mat(T,P,i,j)

    adj2 = np.zeros((T.shape[0] + P.shape[0] , T.shape[0] + P.shape[0]))
    adj2[i,T.shape[0]:] = 1
    adj2[T.shape[0]:, i] = 1
    adj2[:T.shape[0], j + T.shape[0]] = 1
    adj2[j + T.shape[0], :T.shape[0]] = 1


    # if s == 0:
    #   probing.push(
    #       probes,
    #       specs.Stage.HINT,
    #       next_probe={
    #           'pred_h': probing.strings_pred(T_pos, P_pos),
    #           's': probing.mask_one(s, T.shape[0] + P.shape[0]),
    #           'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
    #           'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
    #           'advance':1,
    #           'i_g':0,
    #           'j_g':0,
    #           's_g':0,
    #           'i_j_mat': i_j_mat,
    #           'adj2':adj2,
    #       })
    # else:
    #   probing.push(
    #       probes,
    #       specs.Stage.HINT,
    #       next_probe={
    #           'pred_h': probing.strings_pred(T_pos, P_pos),
    #           's': probing.mask_one(s, T.shape[0] + P.shape[0]),
    #           'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
    #           'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
    #           'advance':1,
    #           'i_g':0,
    #           'j_g':0,
    #           's_g':1,
    #           'i_j_mat': i_j_mat,
    #           'adj2':adj2
    #       })
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            's': probing.mask_one(s, T.shape[0] + P.shape[0]),
            'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
            'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
            'advance': int(T[i] != P[j]),
            ## equal: one - not equal: 0
            'advance_i':int(T[i] == P[j]), # advance i if they are not equal
            'advance_j':int(T[i] == P[j]), # advance j if they are not equal
            'advance_s':int(T[i] != P[j]), # advacne s if they are not equal
            'i_j_mat': i_j_mat,
            #'adj2':adj2
        })


    while True:
      if T[i] != P[j]:
        break
      elif j == P.shape[0] - 1:
        probing.push(
            probes,
            specs.Stage.OUTPUT,
            next_probe={'match': probing.mask_one(s, T.shape[0] + P.shape[0])
            })
        probing.finalize(probes)
        #debug
        #print("debug advance >> ", debug_advance)
        return s, probes
      else:
        i += 1
        j += 1
        
        i_j_mat = get_i_j_mat(T,P,i,j)

        adj2 = np.zeros((T.shape[0] + P.shape[0] , T.shape[0] + P.shape[0]))
        adj2[i,T.shape[0]:] = 1
        adj2[T.shape[0]:, i] = 1
        adj2[:T.shape[0], j + T.shape[0]] = 1
        adj2[j + T.shape[0], :T.shape[0]] = 1
        #debug
        debug_advance.append(int(T[i] != P[j]))

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pred_h': probing.strings_pred(T_pos, P_pos),
                's': probing.mask_one(s, T.shape[0] + P.shape[0]),
                'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
                'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0]),
                'advance':int(T[i] != P[j]),
                ## equal: one - not equal: 0
                'advance_i':int(T[i] == P[j]), # advance i if they are not equal
                'advance_j':int(T[i] == P[j]), # advance j if they are not equal
                'advance_s':int(T[i] != P[j]), # advacne s if they are not equal
                'i_j_mat': i_j_mat,
                #'adj2':adj2
            })

    s += 1

  # By convention, set probe to head of needle if no match is found
  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={
          'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  #debug
  #print("debug advance >> ", debug_advance)
  return T.shape[0], probes


def kmp_matcher(T: _Array, P: _Array) -> _Out:
  """Knuth-Morris-Pratt string matching (Knuth et al., 1977)."""

  chex.assert_rank([T, P], 1)
  probes = probing.initialize(specs.SPECS['kmp_matcher'])

  T_pos = np.arange(T.shape[0])
  P_pos = np.arange(P.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'string':
              probing.strings_id(T_pos, P_pos),
          'pos':
              probing.strings_pos(T_pos, P_pos),
          'key':
              probing.array_cat(
                  np.concatenate([np.copy(T), np.copy(P)]), _ALPHABET_SIZE),
      })

  pi = np.arange(P.shape[0])
  k = 0

  # Cover the edge case where |P| = 1, and the first half is not executed.
  delta = 1 if P.shape[0] > 1 else 0

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.strings_pred(T_pos, P_pos),
          'pi': probing.strings_pi(T_pos, P_pos, pi),
          'k': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0]),
          'q': probing.mask_one(T.shape[0] + delta, T.shape[0] + P.shape[0]),
          's': probing.mask_one(0, T.shape[0] + P.shape[0]),
          'i': probing.mask_one(0, T.shape[0] + P.shape[0]),
          'phase': 0
      })

  for q in range(1, P.shape[0]):
    while k != pi[k] and P[k] != P[q]:
      k = pi[k]
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.strings_pred(T_pos, P_pos),
              'pi': probing.strings_pi(T_pos, P_pos, pi),
              'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
              'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
              's': probing.mask_one(0, T.shape[0] + P.shape[0]),
              'i': probing.mask_one(0, T.shape[0] + P.shape[0]),
              'phase': 0
          })
    if P[k] == P[q]:
      k += 1
    pi[q] = k
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            'pi': probing.strings_pi(T_pos, P_pos, pi),
            'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
            'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
            's': probing.mask_one(0, T.shape[0] + P.shape[0]),
            'i': probing.mask_one(0, T.shape[0] + P.shape[0]),
            'phase': 0
        })
  q = 0
  s = 0
  for i in range(T.shape[0]):
    if i >= P.shape[0]:
      s += 1
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            'pi': probing.strings_pi(T_pos, P_pos, pi),
            'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
            'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
            's': probing.mask_one(s, T.shape[0] + P.shape[0]),
            'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
            'phase': 1
        })
    while q != pi[q] and P[q] != T[i]:
      q = pi[q]
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.strings_pred(T_pos, P_pos),
              'pi': probing.strings_pi(T_pos, P_pos, pi),
              'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
              'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
              's': probing.mask_one(s, T.shape[0] + P.shape[0]),
              'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
              'phase': 1
          })
    if P[q] == T[i]:
      if q == P.shape[0] - 1:
        probing.push(
            probes,
            specs.Stage.OUTPUT,
            next_probe={'match': probing.mask_one(s, T.shape[0] + P.shape[0])})
        probing.finalize(probes)
        return s, probes
      q += 1

  # By convention, set probe to head of needle if no match is found
  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={
          'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  probing.finalize(probes)

  return T.shape[0], probes
