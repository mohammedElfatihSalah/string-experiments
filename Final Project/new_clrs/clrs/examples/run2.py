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

"""Convert to PyG dataset"""
import time
from absl import app
from absl import flags
from absl import logging

from collections import defaultdict

import numpy as np
import torch

import clrs

from torch_geometric.data import Data, DenseDataLoader
# from torch.utils.data import DataLoader

flags.DEFINE_string('algorithm', 'naive_string_matcher', 'Which algorithm to run.')

FLAGS = flags.FLAGS

def convert_to_dic(sampler, num_samples, batch_size):
    dic = {}
    if num_samples%batch_size==0:
        total_itr = num_samples//batch_size
    else:
        total_itr = num_samples//batch_size + 1
    for total_itr_index in range(total_itr):

        feedback = sampler.next(batch_size)

        for index in range(batch_size):
            dic[total_itr_index*batch_size + index] = {}
            # input features
            for i in range(len(feedback.features) - 1):
                for j in range(len(feedback.features[i])):
                    if i == 1:
                        dic[total_itr_index*batch_size + index][feedback.features[i][j].name] = torch.from_numpy(feedback.features[i][j].data[:,index]).squeeze()
                        #print("feedback.features[i][j]", feedback.features[i][j].data[:,index].shape)
                    else:
                        dic[total_itr_index*batch_size + index][feedback.features[i][j].name] = torch.from_numpy(feedback.features[i][j].data[index]).squeeze()
                        #print("feedback.features[i][j]", feedback.features[i][j])
                # print(torch.tensor(feedback.features.lengths[index]))
                dic[total_itr_index*batch_size + index]["lengths"] = torch.tensor([feedback.features.lengths[index].item()])

            # output features
            for i in range(len(feedback.outputs)):
                # print('feedback.outputs[i].data', feedback.outputs[i].data[index].item())
                dic[total_itr_index*batch_size + index][feedback.outputs[i].name] = torch.tensor([feedback.outputs[i].data[index].item()])
            if total_itr_index*batch_size + index >= num_samples - 1:
                break
    return dic
def convert_to_PyG_data(data_dic):
    PyG_data = []
    for i in range(len(data_dic)):
        PyG_data.append(Data.from_dict(data_dic[i]))
    return PyG_data

def main(unused_argv):
  train_sampler, spec = clrs.clrs21_train(FLAGS.algorithm)
  val_sampler, _ = clrs.clrs21_val(FLAGS.algorithm)
  test_sampler, _ = clrs.clrs21_test(FLAGS.algorithm)
  print("FLAGS.algorithm", FLAGS.algorithm)
  train_data_dic = convert_to_dic(train_sampler, 1000, 64)
  val_data_dic = convert_to_dic(val_sampler, 32, 32)
  test_data_dic = convert_to_dic(test_sampler, 32, 32)
  print('dic', len(train_data_dic))
  print("dic_1", len(val_data_dic))
  print("dic_2", len(test_data_dic))
  PyG_data_train = convert_to_PyG_data(train_data_dic)
  PyG_data_val  = convert_to_PyG_data(val_data_dic)
  PyG_data_test = convert_to_PyG_data(test_data_dic)
  print("PyG_data_train", len(PyG_data_train))
  print(PyG_data_train[0])
  print(PyG_data_val[0])
  print(PyG_data_test[0])
  train_loader = DenseDataLoader(dataset = PyG_data_train, batch_size=32,shuffle=True) 
  for idx, graph in enumerate(train_loader):
      if idx == 0:
          print(graph)
          break



#   dic = {}

#   for step in range(1000):
#     feedback = train_sampler.next(FLAGS.batch_size)
#     feedback = train_sampler.next(1)
#     # print("feedback2", feedback)
#     # print("feedback.features", feedback.outputs)
#     # print("len()feedback.features", len(feedback.features))
#     # print("len(feedback.features[0])", len(feedback.features[0]), len(feedback.features[1]), len(feedback.features[2]))
#     # print("feedback.features[0]", feedback.features[0])
#     # print("feedback.features[1]", feedback.features[1])
#     # print("feedback.features[2]", feedback.features[2])
#     # print("feedback.features.lengths", feedback.features.lengths.size)
#     dic[step] = {}
#     for i in range(len(feedback.features) - 1):
#         for j in range(len(feedback.features[i])):
#             dic[0][feedback.features[i][j].name] = torch.from_numpy(feedback.features[i][j].data)
#         dic[step]["lengths"] = torch.from_numpy(feedback.features.lengths)

#     for i in range(len(feedback.outputs)):
#         # print("feedback.outputs[i].name", feedback.outputs[i].name)
#         # print("feedback.outputs[i].data", feedback.outputs[i].data)
#         dic[step][feedback.outputs[i].name] = torch.from_numpy(feedback.outputs[i].data)

#     # print("dic", dic)
#   print("dic", len(dic))




if __name__ == '__main__':
  app.run(main)
