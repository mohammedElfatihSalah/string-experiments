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

"""JAX implementation of CLRS baseline models."""

import functools
import os
import pickle

from typing import Dict, Tuple, List

import chex
import wandb
import jax
from clrs._src import model
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import wandb
import numpy as np
import haiku as hk


_BIG_NUMBER = 1e5

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_Feedback = samplers.Feedback
_Location = specs.Location
_Seed = jnp.ndarray
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done

@chex.dataclass
class Hint:
  i:chex.Array
  j:chex.Array
  s:chex.Array

@chex.dataclass
class _MessagePassingScanState:
  hint_preds: chex.Array
  diff_logits: chex.Array
  gt_diffs: chex.Array
  output_preds: chex.Array
  hiddens: chex.Array
  prev_hint: Hint
  keys:chex.Array


class Net3(hk.Module):
  """Building blocks (networks) used to encode and decode messages."""

  def __init__(
      self,
      spec: _Spec,
      hidden_dim: int,
      encode_hints: bool,
      decode_hints: bool,
      decode_diffs: bool,
      kind: str,
      inf_bias: bool,
      inf_bias_edge: bool,
      nb_dims,
      pooling:str,
      name: str = 'net',
  ):
    """Constructs a `Net`."""
    super().__init__(name=name)

    self.spec = spec
    self.inf_bias = inf_bias
    self.inf_bias_edge = inf_bias_edge
    self.hidden_dim = hidden_dim
    self.encode_hints = encode_hints
    self.decode_hints = decode_hints
    self.decode_diffs = decode_diffs
    self.kind = kind
    self.nb_dims = nb_dims
    self.pooling = pooling

  def next_hint_byforce(self, advance_s, advance_i, advance_j , prev_hint_, nb_nodes):
    _advance = None

    prev_s = prev_hint_.s
    prev_i = prev_hint_.i
    prev_j = prev_hint_.j

    #print("keys shape >> ", keys.shape)
    #print("prev_i shape >> ", np.expand_dims(prev_i, axis=1).shape)
    #i_key = jnp.matmul(jnp.expand_dims(prev_i, axis=1), keys[:,:,None]).squeeze(-1)

    #print("i_key shape >> ", i_key.shape)
  
    #j_key = jnp.matmul(jnp.expand_dims(prev_j, axis=1), keys[:,:,None]).squeeze(-1)

    
    #advance = (j_key != i_key).astype(int).squeeze(-1)
    
    #print("i_key shape >> ", i_key.shape)
    #_advance = prev_j.argmax(-1)
    #print("_advance shape >> ", _advance.shape)
    
    bs, nb_nodes = prev_s.shape

    # === advance shape (batch_size,) === #
    # updating "s" according to "advance" value
    # s is advanced by one step if the corresponding "advance" 
    # is one, otherwise, it stays the same
    # s shape (batch_size, nb_nodes)
    idx = jnp.argmax(prev_s, axis=-1) + 1
    idx = hk.one_hot(idx, nb_nodes)
    new_s = jnp.where(advance[:,None] == 1, idx, prev_s)

    # updating "i" according to "advance" value
    # i is advanced by one step if corresponding "advancd" is zero
    # otherwise it's reset to the position of s + 1.
    # i shape (batch_size, nb_nodes)
    idx = jnp.argmax(prev_i, axis=-1) + 1
    idx = hk.one_hot(idx, nb_nodes)
    new_i = jnp.where(advance[:,None] != 1, idx, new_s)

    # updating "j" according to "advance" value
    # j is advanced if the corresponding "advance" is zero
    # otherwise it's reset to the starting of the pattern
    # j shape (batch_size, nb_nodes)

    print("start of pattern >> ", 4* nb_nodes // 5 )
    idx = jnp.argmax(prev_j, axis=-1) + 1
    idx = hk.one_hot(idx, nb_nodes)
    start_of_pattern_idx = 4* nb_nodes // 5
    reset_j = np.zeros((bs, nb_nodes))
    reset_j[:, start_of_pattern_idx] = 1
    new_j = jnp.where(advance[:,None] != 1, idx, reset_j)

    prev_hint_.s = new_s
    prev_hint_.i = new_i
    prev_hint_.j = new_j

    return prev_hint_, _advance

  def _next_advance(self, prev_hint_, keys):
    advance = None
    prev_s = prev_hint_.s
    prev_i = prev_hint_.i
    prev_j = prev_hint_.j
    i_key = jnp.matmul(jnp.expand_dims(prev_i, axis=1), keys[:,:,None]).squeeze(-1)
    j_key = jnp.matmul(jnp.expand_dims(prev_j, axis=1), keys[:,:,None]).squeeze(-1)
    advance = (j_key != i_key).astype(int).squeeze(-1)
    return advance


  def _msg_passing_step(self,
                        mp_state: _MessagePassingScanState,
                        i: int,
                        hints: List[_DataPoint],
                        repred: bool,
                        lengths: chex.Array,
                        nb_nodes: int,
                        inputs: _Trajectory,
                        pooling:str,
                        first_step: bool = False,
                        ):
    prev_hint_ = mp_state.prev_hint
    keys = mp_state.keys

    _advance = None
    #print("keys >> ", keys)
    if (not first_step) and repred and self.decode_hints:
      decoded_hint = _decode_from_preds(self.spec, mp_state.hint_preds)
      advance = decoded_hint['advance'].data
      
      prev_hint_, _ = self.next_hint_byforce(advance, prev_hint_, nb_nodes)
      _advance   = self._next_advance(prev_hint_, keys)

      cur_hint = []
      exp_hints = ['s', 'i', 'j']
      for hint_name in exp_hints:
        _, loc, typ = self.spec[hint_name]
        cur_hint.append(
            probing.DataPoint(
                name=hint_name, location=loc, type_=typ, data=prev_hint_[hint_name]))
     
      
      for hint in decoded_hint:
        if hint in exp_hints:
          continue
        cur_hint.append(decoded_hint[hint])
    else:
      cur_hint = []
      for hint in hints:
        # save the current value of i, j, s 
        if hint.name == 'i':
          prev_hint_.i = hint.data[i]
        if hint.name == 's':
          prev_hint_.s = hint.data[i]
        if hint.name == 'j':
          prev_hint_.j = hint.data[i]
        _, loc, typ = self.spec[hint.name]
        cur_hint.append(
            probing.DataPoint(
                name=hint.name, location=loc, type_=typ, data=hint.data[i]))
      #prev_hint_, _advance = self.next_hint_byforce(keys, prev_hint_, nb_nodes)

      _advance = self._next_advance(prev_hint_, keys)

    gt_diffs = None
    if hints[0].data.shape[0] > 1 and self.decode_diffs:
      gt_diffs = {
          _Location.NODE: jnp.zeros((self.batch_size, nb_nodes)),
          _Location.EDGE: jnp.zeros((self.batch_size, nb_nodes, nb_nodes)),
          _Location.GRAPH: jnp.zeros((self.batch_size))
      }
      for hint in hints:
        hint_cur = jax.lax.dynamic_index_in_dim(hint.data, i, 0, keepdims=False)
        hint_nxt = jax.lax.dynamic_index_in_dim(
            hint.data, i+1, 0, keepdims=False)
        if len(hint_cur.shape) == len(gt_diffs[hint.location].shape):
          hint_cur = jnp.expand_dims(hint_cur, -1)
          hint_nxt = jnp.expand_dims(hint_nxt, -1)
        gt_diffs[hint.location] += jnp.any(hint_cur != hint_nxt, axis=-1)
      for loc in _Location:
        gt_diffs[loc] = (gt_diffs[loc] > 0.0).astype(jnp.float32) * 1.0
    
    hiddens, output_preds_cand, hint_preds, diff_logits = self._one_step_pred(
        inputs, cur_hint, mp_state.hiddens, nb_nodes, pooling=pooling, prev_hint=prev_hint_)

    if first_step:
      output_preds = output_preds_cand
      
    else:
      output_preds = {}
      for outp in mp_state.output_preds:
        is_not_done = _is_not_done_broadcast(lengths, i,
                                             output_preds_cand[outp])
        output_preds[outp] = is_not_done * output_preds_cand[outp] + (
            1.0 - is_not_done) * mp_state.output_preds[outp]

    if self.decode_diffs:
      if self.decode_hints:
        if hints[0].data.shape[0] == 1 or repred:
          diff_preds = {}
          for loc in _Location:
            diff_preds[loc] = (diff_logits[loc] > 0.0).astype(jnp.float32) * 1.0
        else:
          diff_preds = gt_diffs
        for hint in hints:
          #######################################
          if hint.name in ['s', 'i', 'j']:
            continue
          prev_hint = (
              hint.data[0]
              if first_step else mp_state.hint_preds[hint.name])
          if first_step and hint.type_ == _Type.POINTER:
            prev_hint = hk.one_hot(prev_hint, nb_nodes)
          cur_diffs = diff_preds[hint.location]
          while len(prev_hint.shape) > len(cur_diffs.shape):
            cur_diffs = jnp.expand_dims(cur_diffs, -1)
          hint_preds[hint.name] = (
              cur_diffs * hint_preds[hint.name] + (1.0 - cur_diffs) * prev_hint)

    
    # if _advance != None:
    #   print("here")
    #   hint_preds['advance'] = _advance.astype(float)

    #print(f"hint preds for step {i}>> ", hint_preds['advance'])
    new_mp_state = _MessagePassingScanState(
        hint_preds=hint_preds, diff_logits=diff_logits, gt_diffs=gt_diffs,
        output_preds=output_preds, hiddens=hiddens, prev_hint=prev_hint_, keys=keys)
    # Complying to jax.scan, the first returned value is the state we carry over
    # the second value is the output that will be stacked over steps.
    return new_mp_state, new_mp_state

  def __call__(self, features: _Features, repred: bool):
    """Network inference step."""
    inputs = features.inputs
    hints = features.hints
    lengths = features.lengths
    
    # getting keys input
    keys = None
    for inp in inputs:
      if inp.name == 'key':
        keys = inp.data.argmax(-1)
        break

    for inp in inputs:
      if inp.location in [_Location.NODE, _Location.EDGE]:
        self.batch_size = inp.data.shape[0]
        nb_nodes = inp.data.shape[1]
        break

    self._construct_encoders_decoders()
    self._construct_processor()

    nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
    hiddens = jnp.zeros((self.batch_size, nb_nodes, self.hidden_dim))

    mp_state = _MessagePassingScanState(
        hint_preds=None, diff_logits=None, gt_diffs=None,
        output_preds=None, hiddens=hiddens, prev_hint =Hint(i=None, j=None, s=None), keys=keys)

    # Do the first step outside of the scan because it has a different
    # computation graph.
    mp_state, _ = self._msg_passing_step(
        mp_state,
        i=0,
        first_step=True,
        hints=hints,
        repred=repred,
        inputs=inputs,
        nb_nodes=nb_nodes,
        lengths=lengths,
        pooling=self.pooling)

    # Then scan through the rest.
    scan_fn = functools.partial(
        self._msg_passing_step,
        first_step=False,
        hints=hints,
        repred=repred,
        inputs=inputs,
        nb_nodes=nb_nodes,
        lengths=lengths,
        pooling=self.pooling)

    _, output_mp_state = hk.scan(
        scan_fn,
        mp_state,
        jnp.arange(nb_mp_steps - 1) + 1,
        length=nb_mp_steps - 1)
    
    
    output_mp_state = jax.tree_multimap(
        lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
        mp_state, output_mp_state)
    #print("output state >> ", output_mp_state.hint_preds)

    def invert(d):
      """Dict of lists -> list of dicts."""
      if d:
        return [dict(zip(d, i)) for i in zip(*d.values())]

    output_preds = invert(output_mp_state.output_preds)
    hint_preds = invert(output_mp_state.hint_preds)
    diff_logits = invert(output_mp_state.diff_logits)
    gt_diffs = invert(output_mp_state.gt_diffs)

    return output_preds[-1], hint_preds, diff_logits, gt_diffs

  def _construct_encoders_decoders(self):
    """Constructs encoders and decoders."""

    # self.lstms = {
    #   'i_g':hk.LSTM(3*self.hidden_dim),
    #   's_g':hk.LSTM(3*self.hidden_dim),
    #   'j_g':hk.LSTM(3*self.hidden_dim),
    #   'advance':hk.LSTM(3*self.hidden_dim),
    # }

    self.enc_inp = {}
    self.dec_out = {}
    if self.encode_hints:
      self.enc_hint = {}
    if self.decode_diffs:
      self.node_dec_diff = hk.Linear(1)
      self.edge_dec_diff = (hk.Linear(1), hk.Linear(1), hk.Linear(1))
      self.graph_dec_diff = (hk.Linear(1), hk.Linear(1))
    if self.decode_hints:
      self.dec_hint = {}

    for name in self.spec:
      stage, loc, t = self.spec[name]
      if stage == _Stage.INPUT:
        self.enc_inp[name] = [hk.Linear(self.hidden_dim)]
        if loc == _Location.EDGE and t == _Type.POINTER:
          # Edge pointers need two-way encoders
          self.enc_inp[name].append(hk.Linear(self.hidden_dim))

      elif stage == _Stage.OUTPUT:
        if loc == _Location.NODE:
          if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            
            self.dec_out[name] = (hk.Linear(1),) 
          elif t == _Type.CATEGORICAL:
            self.dec_out[name] = (hk.Linear(self.nb_dims[name]),)
          elif t == _Type.POINTER:
            self.dec_out[name] = (hk.Linear(self.hidden_dim),
                                  hk.Linear(self.hidden_dim))
          else:
            raise ValueError('Incorrect type')
        elif loc == _Location.EDGE:
          if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            self.dec_out[name] = (hk.Linear(1), hk.Linear(1), hk.Linear(1))
          elif t == _Type.CATEGORICAL:
            cat_dims = self.nb_dims[name]
            self.dec_out[name] = (hk.Linear(cat_dims), hk.Linear(cat_dims),
                                  hk.Linear(cat_dims))
          elif t == _Type.POINTER:
            self.dec_out[name] = (hk.Linear(self.hidden_dim),
                                  hk.Linear(self.hidden_dim),
                                  hk.Linear(self.hidden_dim),
                                  hk.Linear(self.hidden_dim))
          else:
            raise ValueError('Incorrect type')
        elif loc == _Location.GRAPH:
          if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
            self.dec_out[name] = (hk.Linear(1), hk.Linear(1))
          elif t == _Type.CATEGORICAL:
            cat_dims = self.nb_dims[name]
            self.dec_out[name] = (hk.Linear(cat_dims), hk.Linear(cat_dims))
          elif t == _Type.POINTER:
            self.dec_out[name] = (hk.Linear(self.hidden_dim),
                                  hk.Linear(self.hidden_dim),
                                  hk.Linear(self.hidden_dim))
          else:
            raise ValueError('Incorrect type')
        else:
          raise ValueError('Incorrect location')

      elif stage == _Stage.HINT:
        if self.encode_hints:
          self.enc_hint[name] = [hk.Linear(self.hidden_dim)]
          if loc == _Location.EDGE and t == _Type.POINTER:
            # Edge pointers need two-way encoders
            self.enc_hint[name].append(hk.Linear(self.hidden_dim))

        if self.decode_hints:
          if loc == _Location.NODE:
            if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
              self.dec_hint[name] = (hk.Linear(1),)
            elif t == _Type.CATEGORICAL:
              self.dec_hint[name] = (hk.Linear(self.nb_dims[name]),)
            elif t == _Type.POINTER:
              self.dec_hint[name] = (hk.Linear(self.hidden_dim),
                                     hk.Linear(self.hidden_dim))
            else:
              raise ValueError('Incorrect type')
          elif loc == _Location.EDGE:
            if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
              self.dec_hint[name] = (hk.Linear(1), hk.Linear(1), hk.Linear(1))
            elif t == _Type.CATEGORICAL:
              cat_dims = self.nb_dims[name]
              self.dec_hint[name] = (hk.Linear(cat_dims), hk.Linear(cat_dims),
                                     hk.Linear(cat_dims))
            elif t == _Type.POINTER:
              self.dec_hint[name] = (hk.Linear(self.hidden_dim),
                                     hk.Linear(self.hidden_dim),
                                     hk.Linear(self.hidden_dim),
                                     hk.Linear(self.hidden_dim))
            else:
              raise ValueError('Incorrect type')
          elif loc == _Location.GRAPH:
            if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
              self.dec_hint[name] = (hk.Linear(1), hk.Linear(1))
            elif t == _Type.CATEGORICAL:
              cat_dims = self.nb_dims[name]
              self.dec_hint[name] = (hk.Linear(cat_dims), hk.Linear(cat_dims))
            elif t == _Type.POINTER:
              self.dec_hint[name] = (hk.Linear(self.hidden_dim),
                                     hk.Linear(self.hidden_dim),
                                     hk.Linear(self.hidden_dim))
            else:
              raise ValueError('Incorrect type')
          else:
            raise ValueError('Incorrect location')

  def _construct_processor(self):
    """Constructs processor."""
    if self.kind in ['deepsets', 'mpnn', 'pgn']:
      self.mpnn = processors.MPNN(
          out_size=self.hidden_dim,
          mid_act=jax.nn.relu,
          activation=jax.nn.relu,
          reduction=jnp.max,
          msgs_mlp_sizes=[
              self.hidden_dim,
              self.hidden_dim,
          ])
    elif self.kind == 'gat':
      self.mpnn = processors.GAT(
          out_size=self.hidden_dim,
          nb_heads=1,
          activation=jax.nn.relu,
          residual=True)
    elif self.kind in ['mpnn_multi', 'pgn_multi']:
      self.mpnn = processors.MPNN_Multilayer(
          num_layers=2,
          out_size=self.hidden_dim,
          mid_act=jax.nn.relu,
          activation=jax.nn.relu,
          reduction=jnp.max,
          msgs_mlp_sizes=[
              self.hidden_dim,
              self.hidden_dim,
          ])
    elif self.kind == 'gat_multi':
      self.mpnn = processors.GAT_Multilayer(
          num_layers=2,
          out_size=self.hidden_dim,
          nb_heads=1,
          activation=jax.nn.relu,
          residual=True,
      )

  
  def lstm_graph_pool(
    self,
    hint_name:str,
    lstms:dict,
    h_t:_Array, 
    T:int
  )-> _Array:

    assert len(list(h_t.shape)) == 3, 'should be three dimensions'
    assert T > 0, 'at least one iteration'
    
    bs, nb_nodes, hidden_dim = h_t.shape
    lstm = lstms[hint_name]
    q, o = self.initialize_query_and_o(bs, hidden_dim)
    state = lstm.initial_state(bs)
    o_history = []
    for i in range(T): 
      q, state = self.feed_lstm(lstm, o, state)
      a = self.compute_attention_weight(h_t, q)
      o = self.compute_o(h_t, a)
      o_history.append(o)
    gr_emb = self.aggr_o(o_history)
    assert gr_emb.shape == (bs,hidden_dim,)
    return gr_emb

  def initialize_query_and_o(self, bs:int, hidden_dim:int):
    q = jnp.zeros((bs,hidden_dim))
    o = jnp.zeros((bs, hidden_dim))
    return q, o
  
  def feed_lstm(self, lstm, q, state):
    output, next_state = lstm(q, state)
    return output, next_state

  def compute_attention_weight(self, h_t, q):
    b, n, hid_dim = h_t.shape
    assert q.shape == (b,hid_dim)
    e = jnp.matmul(h_t, jnp.expand_dims(q, axis=-1)).squeeze(-1) 
    assert e.shape == (b, n)
    m = jnp.max(e, -1)
    m = m.reshape(b,1)
    e = e - m
    e = jnp.exp(e)
    a = jax.nn.softmax(e , axis = -1)
    assert a.shape == (b, n)
    return a 
  
  def compute_o(self, h_t, a):
    b, n, hid_dim = h_t.shape
    o = jnp.matmul(jnp.expand_dims(a, axis=1), h_t).squeeze(1)
    assert o.shape == (b, hid_dim)
    return o
  
  def aggr_o(self, o_history):
    result = jnp.stack(
      o_history,
      axis = 1
    )
    result = jnp.max(result, axis=1)
    return result

  def _one_step_pred(
      self,
      inputs: _Trajectory,
      hints: _Trajectory,
      hidden: _Array,
      nb_nodes: int,
      pooling:str,
      prev_hint
  ):
    """Generates one step predictions."""

    bs = hidden.shape[0]

    node_fts = jnp.zeros((self.batch_size, nb_nodes, self.hidden_dim))
    edge_fts = jnp.zeros((self.batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = jnp.zeros((self.batch_size, self.hidden_dim))
    adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), self.batch_size, axis=0)
    # added mat
    mat = None
    for inp in inputs:
      if inp.name =='mat':
        mat = inp.data
      # Extract shared logic with hints and loss
      encoder = self.enc_inp[inp.name][0]
      if inp.type_ == _Type.POINTER:
        in_data = hk.one_hot(inp.data, nb_nodes)
      else:
        in_data = inp.data.astype(jnp.float32)
      if inp.type_ == _Type.CATEGORICAL:
        encoding = encoder(in_data)
      else:
        encoding = encoder(jnp.expand_dims(in_data, -1))
      if inp.location == _Location.NODE:
        if inp.type_ == _Type.POINTER:
          edge_fts += encoding
          adj_mat += ((in_data + jnp.transpose(in_data, (0, 2, 1))) >
                      0.0).astype('float32')
        else:
          node_fts += encoding
      elif inp.location == _Location.EDGE:
        if inp.type_ == _Type.POINTER:
          # Aggregate pointer contributions across sender and receiver nodes
          encoding_2 = self.enc_inp[inp.name][1](jnp.expand_dims(in_data, -1))
          edge_fts += jnp.mean(encoding, axis=1) + jnp.mean(encoding_2, axis=2)
        else:
          edge_fts += encoding
          if inp.type_ == _Type.MASK:
            adj_mat += (in_data > 0.0).astype('float32')
      elif inp.location == _Location.GRAPH:
        if inp.type_ == _Type.POINTER:
          node_fts += encoding
        else:
          graph_fts += encoding

    if self.encode_hints:
      for hint in hints:
        encoder = self.enc_hint[hint.name][0]
        if hint.type_ == _Type.POINTER:
          in_data = hk.one_hot(hint.data, nb_nodes)
        else:
          in_data = hint.data.astype(jnp.float32)
        if hint.type_ == _Type.CATEGORICAL:
          encoding = encoder(in_data)
        else:
          encoding = encoder(jnp.expand_dims(in_data, -1))
        if hint.location == _Location.NODE:
          if hint.type_ == _Type.POINTER:
            edge_fts += encoding
            adj_mat += ((in_data + jnp.transpose(in_data, (0, 2, 1))) >
                        0.0).astype('float32')
          else:
            node_fts += encoding
        elif hint.location == _Location.EDGE:
          if hint.type_ == _Type.POINTER:
            # Aggregate pointer contributions across sender and receiver nodes
            encoding_2 = self.enc_hint[hint.name][1](
                jnp.expand_dims(in_data, -1))
            edge_fts += jnp.mean(encoding, axis=1) + jnp.mean(
                encoding_2, axis=2)
          else:
            edge_fts += encoding
            if hint.type_ == _Type.MASK:
              adj_mat += (in_data > 0.0).astype('float32')
        elif hint.location == _Location.GRAPH:
          if hint.type_ == _Type.POINTER:
            node_fts += encoding
          else:
            graph_fts += encoding
        else:
          raise ValueError('Invalid hint location')

    if self.kind == 'deepsets':
      adj_mat = jnp.repeat(
          jnp.expand_dims(jnp.eye(nb_nodes), 0), self.batch_size, axis=0)
    elif self.kind == 'mpnn' or self.kind == 'gat' or self.kind == 'mpnn_multi' or self.kind == 'gat_multi':
      adj_mat = mat#jnp.ones_like(adj_mat)
    elif self.kind in ['pgn', 'pgn_multi']:
      adj_mat = (adj_mat > 0.0) * 1.0
    else:
      raise ValueError('Unsupported kind of model')

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    #z = node_fts
    nxt_hidden = self.mpnn(z, edge_fts, graph_fts,
                           (adj_mat > 0.0).astype('float32'))
    h_t = jnp.concatenate([z, nxt_hidden], axis=-1)
    
    hint_preds = {}
    output_preds = {}
    diff_preds = {}

    if self.decode_diffs:
      diff_preds[_Location.NODE] = jnp.squeeze(self.node_dec_diff(h_t), -1)
      e_pred_1 = self.edge_dec_diff[0](h_t)
      e_pred_2 = self.edge_dec_diff[1](h_t)
      e_pred_e = self.edge_dec_diff[2](edge_fts)
      diff_preds[_Location.EDGE] = jnp.squeeze(
          jnp.expand_dims(e_pred_1, -1) + jnp.expand_dims(e_pred_2, -1) +
          e_pred_e, -1)
      gr_emb = jnp.max(h_t, axis=-2)
      g_pred_n = self.graph_dec_diff[0](gr_emb)
      g_pred_g = self.graph_dec_diff[1](graph_fts)
      diff_preds[_Location.GRAPH] = jnp.squeeze(g_pred_n + g_pred_g, -1)
    else:
      diff_preds = {
          _Location.NODE: jnp.ones((self.batch_size, nb_nodes)),
          _Location.EDGE: jnp.ones((self.batch_size, nb_nodes, nb_nodes)),
          _Location.GRAPH: jnp.ones((self.batch_size))
      }

    # decoding the hints 
    if self.decode_hints:
      for hint in hints:
        if hint.name in ['s', 'i', 'j']:
          continue

        decoders = self.dec_hint[hint.name]

        if hint.location == _Location.NODE:
          if hint.type_ in [
              _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
          ]:
            hint_preds[hint.name] = jnp.squeeze(decoders[0](h_t), -1)
          elif hint.type_ == _Type.CATEGORICAL:
            hint_preds[hint.name] = decoders[0](h_t)
          elif hint.type_ == _Type.POINTER:
            p_1 = decoders[0](h_t)
            p_2 = decoders[1](h_t)
            ptr_p = jnp.matmul(p_1, jnp.transpose(p_2, (0, 2, 1)))
            hint_preds[hint.name] = ptr_p
            if self.inf_bias:
              hint_preds[hint.name] -= (1 - adj_mat) * _BIG_NUMBER
          else:
            raise ValueError('Invalid hint type')
        elif hint.location == _Location.EDGE:
          pred_1 = decoders[0](h_t)
          pred_2 = decoders[1](h_t)
          pred_e = decoders[2](edge_fts)
          pred = (
              jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) +
              pred_e)
          if hint.type_ in [
              _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
          ]:
            hint_preds[hint.name] = jnp.squeeze(pred, -1)
          elif hint.type_ == _Type.CATEGORICAL:
            hint_preds[hint.name] = pred
          elif hint.type_ == _Type.POINTER:
            pred_2 = jnp.expand_dims(decoders[3](h_t), -1)
            ptr_p = jnp.matmul(pred, jnp.transpose(pred_2, (0, 3, 2, 1)))
            hint_preds[hint.name] = ptr_p
          else:
            raise ValueError('Invalid hint type')
          if self.inf_bias_edge and hint.type_ in [
              _Type.MASK, _Type.MASK_ONE
          ]:
            hint_preds[hint.name] -= (1 - adj_mat) * _BIG_NUMBER
        elif hint.location == _Location.GRAPH:

          gr_emb = None
          if pooling == 'max':
            print("inside max line: [722] baseline_2.py")
            gr_emb = jnp.max(h_t, axis=-2)
          elif pooling == 'lstm':
            print("inside lstm line: [724] baseline_2.py")
            gr_emb = self.lstm_graph_pool(hint.name, self.lstms, h_t, 3)
          elif pooling == 'mean_i_j':
            print("inside mean_i_j line: [728] baseline_2.py")
            gr_emb_i = jnp.matmul(jnp.expand_dims(prev_hint.i,axis=1), h_t)
            gr_emb_j = jnp.matmul(jnp.expand_dims(prev_hint.j,axis=1), h_t)
            gr_emb = (gr_emb_i + gr_emb_j)/2
            gr_emb = gr_emb.squeeze(axis=1)
          elif pooling == 'max_i_j':
            print("inside max_i_j line: [734] baseline_2.py")
            print("prev_hint shape >> ", prev_hint.i.shape)
            gr_emb_i = jnp.matmul(jnp.expand_dims(prev_hint.i,axis=1), h_t)
            gr_emb_j = jnp.matmul(jnp.expand_dims(prev_hint.j,axis=1), h_t)
            gr_emb =  jnp.maximum(gr_emb_i, gr_emb_j)
            gr_emb = jnp.squeeze(gr_emb, axis=1)
            # squared_diff = (gr_emb_i - gr_emb_j) ** 2
            # gr_emb = jnp.sum(squared_diff, axis = -1)
            #print("baseline_2.py line [741] gr emb shape >> ", gr_emb.shape)

          pred_n = decoders[0](gr_emb)
          #pred_g = decoders[1](graph_fts)
          pred = pred_n + pred_g
          #pred = gr_emb
          if hint.type_ in [
              _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
          ]:
            hint_preds[hint.name] = jnp.squeeze(pred, -1)
          elif hint.type_ == _Type.CATEGORICAL:
            hint_preds[hint.name] = pred
          elif hint.type_ == _Type.POINTER:
            pred_2 = decoders[2](h_t)
            ptr_p = jnp.matmul(
                jnp.expand_dims(pred, 1), jnp.transpose(pred_2, (0, 2, 1)))
            hint_preds[hint.name] = jnp.squeeze(ptr_p, 1)
          else:
            raise ValueError('Invalid hint type')

    


    for out_name in self.dec_out:
      decoders = self.dec_out[out_name]
      _, out_location, out_type = self.spec[out_name]
      if out_location == _Location.NODE:
        if out_type in [
            _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
        ]:
          output_preds[out_name] = jnp.squeeze(decoders[0](h_t), -1)
        elif out_type == _Type.CATEGORICAL:
          output_preds[out_name] = decoders[0](h_t)
        elif out_type == _Type.POINTER:
          p_1 = decoders[0](h_t)
          p_2 = decoders[1](h_t)
          ptr_p = jnp.matmul(p_1, jnp.transpose(p_2, (0, 2, 1)))
          output_preds[out_name] = ptr_p
          if self.inf_bias:
            output_preds[out_name] -= (1 - adj_mat) * _BIG_NUMBER
        else:
          raise ValueError('Invalid output type')
      elif out_location == _Location.EDGE:
        pred_1 = decoders[0](h_t)
        pred_2 = decoders[1](h_t)
        pred_e = decoders[2](edge_fts)
        pred = (
            jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) + pred_e)
        if out_type in [
            _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
        ]:
          output_preds[out_name] = jnp.squeeze(pred, -1)
        elif out_type == _Type.CATEGORICAL:
          output_preds[out_name] = pred
        elif out_type == _Type.POINTER:
          pred_2 = jnp.expand_dims(decoders[3](h_t), -1)
          ptr_p = jnp.matmul(pred, jnp.transpose(pred_2, (0, 3, 2, 1)))
          output_preds[out_name] = ptr_p
        else:
          raise ValueError('Invalid output type')
        if self.inf_bias_edge and out_type in [_Type.MASK, _Type.MASK_ONE]:
          output_preds[out_name] -= (1 - adj_mat) * _BIG_NUMBER
      elif out_location == _Location.GRAPH:
        gr_emb = jnp.max(h_t, axis=-2)
        pred_n = decoders[0](gr_emb)
        pred_g = decoders[1](graph_fts)
        pred = pred_n + pred_g
        if out_type in [
            _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
        ]:
          output_preds[out_name] = jnp.squeeze(pred, -1)
        elif out_type == _Type.CATEGORICAL:
          output_preds[out_name] = pred
        elif out_type == _Type.POINTER:
          pred_2 = decoders[2](h_t)
          ptr_p = jnp.matmul(
              jnp.expand_dims(pred, 1), jnp.transpose(pred_2, (0, 2, 1)))
          output_preds[out_name] = jnp.squeeze(ptr_p, 1)
        else:
          raise ValueError('Invalid output type')
    
   

    return nxt_hidden, output_preds, hint_preds, diff_preds


class BaselineModel3(model.Model):
  """Model implementation with selectable message passing algorithm."""

  def __init__(
      self,
      spec,
      hidden_dim=32,
      kind='mpnn',
      encode_hints=False,
      decode_hints=True,
      decode_diffs=False,
      learning_rate=0.005,
      checkpoint_path='/tmp/clrs3',
      freeze_processor=False,
      dummy_trajectory=None,
      pooling:str  = 'max',
      name='base_model',
  ):
    super(BaselineModel3, self).__init__(spec=spec)

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    self.spec = spec
    self.decode_hints = decode_hints
    self.decode_diffs = decode_diffs
    self.checkpoint_path = checkpoint_path
    self.name = name
    self._freeze_processor = freeze_processor
    self.opt = optax.adam(learning_rate)
    self.pooling = pooling

    if kind == 'pgn_mask':
      inf_bias = True
      inf_bias_edge = True
      kind = 'pgn'
    else:
      inf_bias = False 
      inf_bias_edge = False

    self.nb_dims = {}
    for inp in dummy_trajectory.features.inputs:
      self.nb_dims[inp.name] = inp.data.shape[-1]
    for hint in dummy_trajectory.features.hints:
      self.nb_dims[hint.name] = hint.data.shape[-1]
    for outp in dummy_trajectory.outputs:
      self.nb_dims[outp.name] = outp.data.shape[-1]

    def _use_net(*args, **kwargs):
      return Net3(spec, hidden_dim, encode_hints, decode_hints, decode_diffs,
                 kind, inf_bias, inf_bias_edge, self.nb_dims, pooling=self.pooling)(*args, **kwargs)

    self.net_fn = hk.without_apply_rng(hk.transform(_use_net))
    self.net_fn_apply = jax.jit(self.net_fn.apply, static_argnums=2)
    self.params = None
    self.opt_state = None

  def init(self, features: _Features, seed: _Seed):
    self.params = self.net_fn.init(jax.random.PRNGKey(seed), features, True)
    self.opt_state = self.opt.init(self.params)

  def feedback(self, feedback: _Feedback) -> float:
    """Advance to the next task, incorporating any available feedback."""
    self.params, self.opt_state, cur_loss = self.update(self.params,
                                                        self.opt_state,
                                                        feedback)
    return cur_loss

  def predict(self, features: _Features):
    """Model inference step."""
    outs, hint_preds, diff_logits, gt_diff = self.net_fn_apply(
        self.params, features, True)
    return _decode_from_preds(self.spec,
                              outs), (hint_preds, diff_logits, gt_diff)

  def update(
      self,
      params: hk.Params,
      opt_state: optax.OptState,
      feedback: _Feedback,
  ) -> Tuple[hk.Params, optax.OptState, _Array]:
    """Model update step."""

    def loss(params, feedback):
      (output_preds, hint_preds, diff_logits,
       gt_diffs) = self.net_fn_apply(params, feedback.features, True)

      # f = open("log.txt", "w")
      # f.write(str(hint_preds))
      # f.close()
 

      for inp in feedback.features.inputs:
        if inp.location in [_Location.NODE, _Location.EDGE]:
          nb_nodes = inp.data.shape[1]
          break

      total_loss = 0.0
      lengths = feedback.features.lengths
      if self.decode_diffs:
        for loc in _Location:
          for i in range(len(gt_diffs)):
            is_not_done = _is_not_done_broadcast(
                lengths, i, diff_logits[i][loc])
            diff_loss = (
                jnp.maximum(diff_logits[i][loc], 0) -
                diff_logits[i][loc] * gt_diffs[i][loc] +
                jnp.log1p(jnp.exp(-jnp.abs(diff_logits[i][loc]))) *
                is_not_done)

            total_loss += jnp.mean(diff_loss)

      if self.decode_hints:
        #
        hint_losses = {} 
        for truth in feedback.features.hints:
          if truth.name in ['s', 'i', 'j']:
            continue
          #
          prev_loss = total_loss
          #print("truth data >> ", truth.data)
          for i in range(truth.data.shape[0] - 1):
            assert truth.name in hint_preds[i]
            pred = hint_preds[i][truth.name]
            
            #print("hints pred loss >> ", hint_preds)

            #print("pred shape >> ", pred.primal)
            is_not_done = _is_not_done_broadcast(
                lengths, i, truth.data[i + 1])
            #print("is not done >> ", is_not_done)
            if truth.type_ == _Type.SCALAR:
              if self.decode_diffs:
                total_loss += jnp.mean(
                    (pred - truth.data[i + 1])**2 *
                    gt_diffs[i][truth.location] * is_not_done)
              else:
                total_loss += jnp.mean(
                    (pred - truth.data[i + 1])**2 * is_not_done)
            elif truth.type_ == _Type.MASK:
              if self.decode_diffs:
                loss = jnp.mean(
                    jnp.maximum(pred, 0) - pred * truth.data[i + 1] +
                    jnp.log1p(jnp.exp(-jnp.abs(pred))) *
                    gt_diffs[i][truth.location] * is_not_done)
              else:
                # print("max >> ", jnp.maximum(pred, 0))
                #print("is not done ", is_not_done)
                loss = jnp.mean(
                    jnp.maximum(pred, 0) - pred * truth.data[i] +
                    jnp.log1p(jnp.exp(-jnp.abs(pred))) * is_not_done)
            
              mask = (truth.data != _OutputClass.MASKED.value).astype(
                  jnp.float32)
              
              #print(f"truth {i} is {truth.data[i]} and pred is {pred.primal}" )
              total_loss += jnp.sum(loss*mask)/jnp.sum(mask)
            elif truth.type_ == _Type.MASK_ONE:
              if self.decode_diffs:
                total_loss += jnp.mean(
                    -jnp.sum(
                        truth.data[i + 1] * jax.nn.log_softmax(
                            pred) * is_not_done, axis=-1, keepdims=True) *
                    gt_diffs[i][truth.location])
              else:
                total_loss += jnp.mean(-jnp.sum(
                    truth.data[i + 1] * jax.nn.log_softmax(
                        pred) * is_not_done, axis=-1))
            elif truth.type_ == _Type.CATEGORICAL:
              unmasked_data = truth.data[
                  truth.data == _OutputClass.POSITIVE.value]
              masked_truth = truth.data * (
                  truth.data != _OutputClass.MASKED.value).astype(jnp.float32)
              if self.decode_diffs:
                total_loss += jnp.sum(
                    -jnp.sum(
                        masked_truth[i + 1] * jax.nn.log_softmax(
                            pred), axis=-1, keepdims=True) *
                    jnp.expand_dims(gt_diffs[i][truth.location], -1) *
                    is_not_done) / jnp.sum(unmasked_data)
              else:
                total_loss += jnp.sum(-jnp.sum(
                    masked_truth[i + 1] * jax.nn.log_softmax(pred), axis=-1) *
                                      is_not_done) / jnp.sum(unmasked_data)
            elif truth.type_ == _Type.POINTER:
              if self.decode_diffs:
                total_loss += jnp.mean(-jnp.sum(
                    hk.one_hot(truth.data[i + 1], nb_nodes) *
                    jax.nn.log_softmax(pred),
                    axis=-1) * gt_diffs[i][truth.location] * is_not_done)
              else:
                total_loss += jnp.mean(-jnp.sum(
                    hk.one_hot(truth.data[i + 1], nb_nodes) *
                    jax.nn.log_softmax(pred),
                    axis=-1) * is_not_done)
            else:
              raise ValueError('Incorrect type')
          hint_loss = total_loss - prev_loss
          # TODO: later add primal attricbute
          hint_losses[truth.name] = jnp.asarray((hint_loss / (truth.data.shape[0] - 1))).primal

        total_loss /= (truth.data.shape[0] - 1)  # pylint: disable=undefined-loop-variable

      for truth in feedback.outputs:
        assert truth.name in output_preds
        pred = output_preds[truth.name]
        if truth.type_ == _Type.SCALAR:
          total_loss += jnp.mean((pred - truth.data)**2)
        elif truth.type_ == _Type.MASK:
          loss = (jnp.maximum(pred, 0) - pred * truth.data +
                  jnp.log1p(jnp.exp(-jnp.abs(pred))))
          mask = (truth.data != _OutputClass.MASKED.value).astype(jnp.float32)
          total_loss += jnp.sum(loss*mask)/jnp.sum(mask)
        elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
          unmasked_data = truth.data[truth.data == _OutputClass.POSITIVE.value]
          masked_truth = truth.data * (
              truth.data != _OutputClass.MASKED.value).astype(jnp.float32)
          total_loss += (
              -jnp.sum(masked_truth * jax.nn.log_softmax(pred))
              / jnp.sum(unmasked_data))
        elif truth.type_ == _Type.POINTER:
          total_loss += (
              jnp.mean(-jnp.sum(
                  hk.one_hot(truth.data, nb_nodes) * jax.nn.log_softmax(pred),
                  axis=-1)))
        else:
          raise ValueError('Incorrect type')
      wandb.log(hint_losses)
      return total_loss

    lss, grads = jax.value_and_grad(loss)(params, feedback)
    updates, opt_state = self.opt.update(grads, opt_state)
    if self._freeze_processor:
      params_subset = _filter_processor(params)
      updates_subset = _filter_processor(updates)
      new_params = optax.apply_updates(params_subset, updates_subset)
      new_params = hk.data_structures.merge(params, new_params)
    else:
      new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, lss

  def verbose_loss(self, feedback: _Feedback, extra_info) -> Dict[str, _Array]:
    """Gets verbose loss information."""
    hint_preds, diff_logits, gt_diffs = extra_info

    for inp in feedback.features.inputs:
      if inp.location in [_Location.NODE, _Location.EDGE]:
        nb_nodes = inp.data.shape[1]
        break

    total_loss = 0.0
    lengths = feedback.features.lengths

    losses = {}
    if self.decode_diffs:
      for loc in _Location:
        for i in range(len(gt_diffs)):
          is_not_done = _is_not_done_broadcast(lengths, i, gt_diffs[i][loc])
          diff_loss = (
              jnp.maximum(diff_logits[i][loc], 0) -
              diff_logits[i][loc] * gt_diffs[i][loc] +
              jnp.log1p(jnp.exp(-jnp.abs(diff_logits[i][loc]))) * is_not_done)
          losses[loc.name + '_diff_%d' % i] = jnp.mean(diff_loss)

    if self.decode_hints:
      for truth in feedback.features.hints:
        for i in range(truth.data.shape[0] - 1):
          assert truth.name in hint_preds[i]
          pred = hint_preds[i][truth.name]
          is_not_done = _is_not_done_broadcast(lengths, i, truth.data[i + 1])
          if truth.type_ == _Type.SCALAR:
            if self.decode_diffs:
              total_loss = jnp.mean((pred - truth.data[i + 1])**2 *
                                    gt_diffs[i][truth.location] * is_not_done)
            else:
              total_loss = jnp.mean((pred - truth.data[i + 1])**2 * is_not_done)
          elif truth.type_ == _Type.MASK:
            if self.decode_diffs:
              total_loss = jnp.mean(
                  jnp.maximum(pred, 0) - pred * truth.data[i + 1] +
                  jnp.log1p(jnp.exp(-jnp.abs(pred))) *
                  gt_diffs[i][truth.location] * is_not_done)
            else:
              total_loss = jnp.mean(
                  jnp.maximum(pred, 0) - pred * truth.data[i + 1] +
                  jnp.log1p(jnp.exp(-jnp.abs(pred))) * is_not_done)
          elif truth.type_ == _Type.MASK_ONE:
            if self.decode_diffs:
              total_loss = jnp.mean(
                  -jnp.sum(
                      truth.data[i + 1] * jax.nn.log_softmax(
                          pred) * is_not_done, axis=-1, keepdims=True) *
                  gt_diffs[i][truth.location])
            else:
              total_loss = jnp.mean(-jnp.sum(
                  truth.data[i + 1] * jax.nn.log_softmax(
                      pred) * is_not_done, axis=-1))
          elif truth.type_ == _Type.CATEGORICAL:
            if self.decode_diffs:
              total_loss = jnp.mean(
                  -jnp.sum(
                      truth.data[i + 1] * jax.nn.log_softmax(
                          pred), axis=-1, keepdims=True) *
                  jnp.expand_dims(gt_diffs[i][truth.location], -1) *
                  is_not_done)
            else:
              total_loss = jnp.mean(-jnp.sum(
                  truth.data[i + 1] * jax.nn.log_softmax(pred), axis=-1) *
                                    is_not_done)
          elif truth.type_ == _Type.POINTER:
            if self.decode_diffs:
              total_loss = jnp.mean(-jnp.sum(
                  hk.one_hot(truth.data[i + 1], nb_nodes) *
                  jax.nn.log_softmax(pred),
                  axis=-1) * gt_diffs[i][truth.location] * is_not_done)
            else:
              total_loss = jnp.mean(-jnp.sum(
                  hk.one_hot(truth.data[i + 1], nb_nodes) *
                  jax.nn.log_softmax(pred),
                  axis=-1) * is_not_done)
          else:
            raise ValueError('Incorrect type')
          losses[truth.name + '_%d' % i] = total_loss
    return losses

  def restore_model(self, file_name: str, only_load_processor: bool = False):
    """Restore model from `file_name`."""
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      if only_load_processor:
        restored_params = _filter_processor(restored_state['params'])
      else:
        restored_params = restored_state['params']
      self.params = hk.data_structures.merge(self.params, restored_params)
      self.opt_state = restored_state['opt_state']

  def save_model(self, file_name: str):
    """Save model (processor weights only) to `file_name`."""
    os.makedirs(self.checkpoint_path, exist_ok=True)
    to_save = {'params': self.params, 'opt_state': self.opt_state}
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'wb') as f:
      pickle.dump(to_save, f)


def _decode_from_preds(spec: _Spec, preds: _Array) -> Dict[str, _DataPoint]:
  """Decodes outputs using appropriate functions depending on algorithm spec."""
  result = {}
  for name in preds.keys():
    _, loc, typ = spec[name]
    data = preds[name]
    if typ == _Type.SCALAR:
      pass
    elif typ == _Type.MASK:
      data = (data > 0.0) * 1.0
    elif typ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
      cat_size = data.shape[-1]
      best = jnp.argmax(data, -1)
      data = hk.one_hot(best, cat_size)
    elif typ == _Type.POINTER:
      data = jnp.argmax(data, -1)
    else:
      raise ValueError('Invalid type')
    result[name] = probing.DataPoint(
        name=name, location=loc, type_=typ, data=data)
  return result


def _filter_processor(params: hk.Params) -> hk.Params:
  return hk.data_structures.filter(
      lambda module_name, n, v: 'construct_processor' in module_name, params)
