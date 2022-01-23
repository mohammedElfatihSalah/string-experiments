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

"""Algorithm specs.

The "spec" of each algorithm is a static set of `(stage, loc, type)`-tuples.

- `stage`: One of either an `input`, `output` or `hint`
- `location`: Each datum is associated with either the `node`, `edge` or `graph`
- `type`: Either a `scalar`, `categorical`, `mask`, `mask_one` or `pointer`

The dataflow for an algorithm is represented by `(stage, loc, type, data)`
"probes" that are valid under that algorithm's spec. It contains a single
snapshot for each `input` and `output` and a time-series of intermediate
algorithmic states (`hint`).

At minimum, each node contains a `pos` probe that serves as a unique index e.g.
for representing sequential data where appropriate
"""

import enum
import types
from typing import Dict, Tuple


class _OrderedEnum(enum.Enum):

  def __lt__(self, other):
    assert self.__class__ is other.__class__
    return self.value < other.value  # pylint: disable=comparison-with-callable


class Stage(_OrderedEnum):
  INPUT = 'input'
  OUTPUT = 'output'
  HINT = 'hint'


class Location(_OrderedEnum):
  NODE = 'node'
  EDGE = 'edge'
  GRAPH = 'graph'


class Type(_OrderedEnum):
  SCALAR = 'scalar'
  CATEGORICAL = 'categorical'
  MASK = 'mask'
  MASK_ONE = 'mask_one'
  POINTER = 'pointer'


class OutputClass(_OrderedEnum):
  POSITIVE = 1
  NEGATIVE = 0
  MASKED = -1

Spec = Dict[str, Tuple[Stage, Location, Type]]

CLRS_21_ALGS = [
    'bellman_ford',
    'bfs',
    'binary_search',
    'bubble_sort',
    'dag_shortest_paths',
    'dfs',
    'dijkstra',
    'find_maximum_subarray_kadane',
    'floyd_warshall',
    'heapsort',
    'insertion_sort',
    'kmp_matcher',
    'matrix_chain_order',
    'minimum',
    'mst_prim',
    'naive_string_matcher',
    'optimal_bst',
    'quickselect',
    'quicksort',
    'task_scheduling',
    'topological_sort',
    'parallel_string_matcher'
]

SPECS = types.MappingProxyType({
    'insertion_sort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'bubble_sort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'heapsort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'parent': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'largest': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'heap_size': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'quicksort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'p': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'r': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'quickselect': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'median': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'p': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'r': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i_rank': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'target': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'minimum': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'min': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'min_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'binary_search': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'target': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'return': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'find_maximum_subarray': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'start': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'right_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'right_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'right_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'cross_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'cross_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'cross_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'ret_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'ret_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'ret_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'left_x_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'right_x_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'find_maximum_subarray_kadane': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'start': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'best_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'best_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'best_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'sum': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'matrix_chain_order': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'p': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'm': (Stage.HINT, Location.EDGE, Type.SCALAR),
        's_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK)
    },
    'lcs_length': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'b': (Stage.OUTPUT, Location.EDGE, Type.CATEGORICAL),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'b_h': (Stage.HINT, Location.EDGE, Type.CATEGORICAL),
        'c': (Stage.HINT, Location.EDGE, Type.SCALAR)
    },
    'optimal_bst': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'p': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'q': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'root': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'root_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'e': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'w': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK)
    },
    'activity_selector': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'f': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'selected_h': (Stage.HINT, Location.NODE, Type.MASK),
        'm': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'task_scheduling': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'd': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'w': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'selected_h': (Stage.HINT, Location.NODE, Type.MASK),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        't': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'dfs': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'topological_sort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'topo': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'topo_head': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'topo_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'topo_head_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'strongly_connected_components': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'scc_id': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'scc_id_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'A_t': (Stage.HINT, Location.EDGE, Type.MASK),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },
    'articulation_points': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'is_cut': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'is_cut_h': (Stage.HINT, Location.NODE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        'low': (Stage.HINT, Location.NODE, Type.SCALAR),
        'child_cnt': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'bridges': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'is_bridge': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'is_bridge_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        'low': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'bfs': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'reach_h': (Stage.HINT, Location.NODE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER)
    },
    'mst_kruskal': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'in_mst': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'in_mst_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'root_u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'root_v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mask_u': (Stage.HINT, Location.NODE, Type.MASK),
        'mask_v': (Stage.HINT, Location.NODE, Type.MASK),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'mst_prim': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'key': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'bellman_ford': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'msk': (Stage.HINT, Location.NODE, Type.MASK)
    },
    'dag_shortest_paths': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'topo_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'topo_head_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },
    'dijkstra': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'floyd_warshall': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'Pi': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'Pi_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'D': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'bipartite_matching': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        't': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'in_matching': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'in_matching_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'A_h': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'adj_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'msk': (Stage.HINT, Location.NODE, Type.MASK),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },

    'parallel_string_matcher': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end_match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'bipartite': (Stage.HINT, Location.EDGE, Type.CATEGORICAL), 
        'everything_matched':(Stage.HINT, Location.NODE, Type.MASK),
        't': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },

    
    # 'naive_string_matcher': {
    #     'string': (Stage.INPUT, Location.NODE, Type.MASK),

    #     'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
    #     'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
    #     'pattern_start':(Stage.INPUT, Location.NODE, Type.MASK_ONE),
    #     'predecessor': (Stage.INPUT, Location.EDGE, Type.MASK),
    #     # added mat
    #     #'mat': (Stage.INPUT, Location.EDGE, Type.MASK), 
    #     #added_mat_i_j
    #     #'mat_i_j':(Stage.HINT, Location.EDGE, Type.MASK),
    #     'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
    #     #'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
    #     # commented s
    #     's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    #     'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    #     'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    #     'adj':(Stage.HINT, Location.EDGE, Type.MASK),
    # },
  
    'naive_string_matcher': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'predecessor': (Stage.INPUT, Location.EDGE, Type.MASK),
        #'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        
        'match': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
        #'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        #'i_g':(Stage.HINT, Location.GRAPH, Type.MASK),
        #'j_g':(Stage.HINT, Location.GRAPH, Type.MASK),
        #'s_g':(Stage.HINT, Location.GRAPH, Type.MASK),
        'advance':(Stage.HINT, Location.GRAPH, Type.MASK),
        #'advance_i':(Stage.HINT, Location.GRAPH, Type.MASK),
        #'advance_j':(Stage.HINT, Location.GRAPH, Type.MASK),
        #'advance_s':(Stage.HINT, Location.GRAPH, Type.MASK),
         #'i_j_mat': (Stage.HINT, Location.EDGE, Type.MASK),
         #'adj2': (Stage.HINT, Location.EDGE, Type.MASK),
    },
    
    'kmp_matcher': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'q': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },

    'segments_intersect': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'intersect': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'dir': (Stage.HINT, Location.NODE, Type.SCALAR),
        'on_seg': (Stage.HINT, Location.NODE, Type.MASK)
    },
    'graham_scan': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'best': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'atans': (Stage.HINT, Location.NODE, Type.SCALAR),
        'in_hull_h': (Stage.HINT, Location.NODE, Type.MASK),
        'stack_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        'last_stack': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'jarvis_march': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'in_hull_h': (Stage.HINT, Location.NODE, Type.MASK),
        'best': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'last_point': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'endpoint': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    }
})
