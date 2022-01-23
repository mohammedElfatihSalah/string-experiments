from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
from clrs._src.algorithms.strings_graph_structures import get_t, get_bipartite_mat, get_everything_matched_to_this_point
import numpy as np


_Array = np.ndarray
_Out = Tuple[int, probing.ProbesDict]

_ALPHABET_SIZE = 4

def parallel_string_matcher(T: _Array, P: _Array) -> _Out:
  """Naive string matching."""

  chex.assert_rank([T, P], 1)
  probes = probing.initialize(specs.SPECS['parallel_string_matcher'])

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

  s = 0
 
  while s <= T.shape[0] - P.shape[0]:
    i = s
    j = 0

    bipartite = get_bipartite_mat(np.copy(T), np.copy(P), s)
    t = get_t(np.copy(T), np.copy(P), s)
    #
    t_in_pattern = (s-t) + T.shape[0] - 1
    everything_matched = get_everything_matched_to_this_point(np.copy(T), np.copy(P),s)

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            's': probing.mask_one(s, T.shape[0] + P.shape[0]),
            'bipartite': bipartite, 
            't': probing.mask_one(t, T.shape[0] + P.shape[0]),
            'everything_matched': everything_matched,
        })

    while True:
      if T[i] != P[j]:
        break
      elif j == P.shape[0] - 1:
        t = get_t(np.copy(T), np.copy(P), s)
        t = (s-t) + T.shape[0] - 1
        probing.push(
            probes,
            specs.Stage.OUTPUT,
            next_probe={
              'match': probing.mask_one(s, T.shape[0] + P.shape[0]),
              'end_match': probing.mask_one(t, T.shape[0] + P.shape[0]),
              })
        probing.finalize(probes)
        return s, probes
      else:
        i += 1
        j += 1
        
    s += 1

  # By convention, set probe to head of needle if no match is found
  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={
          'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  return T.shape[0], probes


