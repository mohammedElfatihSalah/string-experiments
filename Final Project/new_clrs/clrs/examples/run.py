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

"""Run a full test run for one or more algorithmic tasks from CLRS."""
import wandb
import time
import logging as lg
from absl import app
from absl import flags
from absl import logging

import numpy as np

import clrs


flags.DEFINE_string('algorithm', 'naive_string_matcher', 'Which algorithm to run.')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_integer('batch_size',32 * 2, 'Batch size used for training.')
flags.DEFINE_integer('train_steps', 80000, 'Number of training steps.')
flags.DEFINE_integer('log_every', 10, 'Logging frequency.')
flags.DEFINE_boolean('verbose_logging', False, 'Whether to log aux losses.')

flags.DEFINE_integer('hidden_size', 32,
                   'Number of hidden size units of the model.')
# changed .003 -> .0003
flags.DEFINE_float('learning_rate', 0.003, 'Learning rate to use.')

flags.DEFINE_boolean('encode_hints', True,
                     'Whether to provide hints as model inputs.')
flags.DEFINE_boolean('decode_hints', True,
                     'Whether to provide hints as model outputs.')
flags.DEFINE_boolean('decode_diffs', False,
                     'Whether to predict masks within the model.')
flags.DEFINE_enum('processor_type', 'pgn',
                  ['deepsets', 'mpnn', 'pgn', 'gat','mpnn_multi', 'pgn_multi', 'gat_multi'],
                  'Whether to predict masks within the model.')

flags.DEFINE_string('checkpoint_path', '/tmp/clrs3',
                    'Path in which checkpoints are saved.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS



def compute_average_metric(feedback, aux, hint_name):
  ''' 
  a method to compute the average accuracy for a given hint,
  the hint must be of type  >> MASK << , and location >> GRAPH << 
  '''
  
  all_hints_pred = aux[0]

  hint_len = np.array(feedback.features.lengths, dtype=np.int32) 
  time_steps = len(all_hints_pred)

  bs = all_hints_pred[0][hint_name].shape[0]

  hint_pred = np.zeros((time_steps, bs))
  for i in range(time_steps):
    hint_pred[i] = all_hints_pred[i][hint_name]
  
  #print("hint pred >> ", hint_pred)
 
  
  hint_gt = None
  hints = feedback.features[1]
  for hint in hints:
    if hint.name == hint_name:
      hint_gt = hint.data
  #print("gt >> ", hint_gt)
  #print("gt shape >> ", hint_gt.shape)

  gt_idx = hint_len - 1
  preds_idx = gt_idx - 1
  accuracy = 0
  avg_shift = 0

  #print("hint len >> ", hint_len)
  for i in range(bs):
    idx = hint_len[i]
    sample_gt   = hint_gt[:idx-1,i] 
    sample_pred = hint_pred[:idx-1,i] 
    accuracy += np.mean(sample_gt == (sample_pred > 0.0))
    nb_shift_pred = np.sum((sample_pred > 0.0))
    nb_shift_gt   = np.sum(sample_gt)
    avg_shift += np.abs(nb_shift_gt - nb_shift_pred)
  
  accuracy /= bs
  avg_shift /= bs
  return accuracy, avg_shift

def main(unused_argv):
  run_name = input("Enter a name for your run >> ")
  wandb.init(project="String Algorithms", entity='arabkhla')
  wandb.run.name = run_name
  wandb.save()
  # Use canonical CLRS-21 samplers.
  clrs21_spec = clrs.CLRS21
  logging.info('Using CLRS21 spec: %s', clrs21_spec)
  train_sampler, spec = clrs.clrs21_train(FLAGS.algorithm)
#   print("train")
#   print("train_sampler", train_sampler)
#   print(spec)
#   print("val")


  val_sampler, _ = clrs.clrs21_val(FLAGS.algorithm)

  model = clrs.models.BaselineModel(
      spec=spec,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=FLAGS.encode_hints,
      decode_hints=FLAGS.decode_hints,
      decode_diffs=FLAGS.decode_diffs,
      kind=FLAGS.processor_type,
      learning_rate=FLAGS.learning_rate,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dummy_trajectory=train_sampler.next(FLAGS.batch_size),
      #pooling ='max_i_j'
  )

  # log hyperparameters
  config = wandb.config
  config.learning_rate = FLAGS.learning_rate
  config.hidden_size = FLAGS.hidden_size
  config.processor_type = FLAGS.processor_type
  config.freeze_processor = FLAGS.freeze_processor
  config.batch_size = FLAGS.batch_size
  config.algorithm = FLAGS.algorithm

  def evaluate(step, model, feedback, extras=None, verbose=False):
    """Evaluates a model on feedback."""
    examples_per_step = len(feedback.features.lengths)
    out = {'step': step, 'examples_seen': step * examples_per_step}
    predictions, aux = model.predict(feedback.features)
    out.update(clrs.evaluate(feedback, predictions))
    if extras:
      out.update(extras)
    if verbose:
      out.update(model.verbose_loss(feedback, aux))

    def unpack(v):
      try:
        return v.item()  # DeviceArray
      except AttributeError:
        return v
    
    accuracy, shift = compute_average_metric(feedback, aux, 'advance')
    return {k: unpack(v) for k, v in out.items()}, accuracy, shift 

  # Training loop.
  best_score = 0.
  best_average_accuracy = 0.
  best_average_shift = 1e5

  for step in range(FLAGS.train_steps):
    feedback = train_sampler.next(FLAGS.batch_size)
    #print("feedback.features >> ", feedback.features[0][2].data)

    # Initialize model.
    if step == 0:
      t = time.time()
      model.init(feedback.features, FLAGS.seed)

    # Training step step.
    cur_loss = model.feedback(feedback)
    
    if step == 0:
      logging.info('Compiled feedback step in %f s.', time.time() - t)

    # Periodically evaluate model.
    if step % FLAGS.log_every == 0:
      # Training info.
      train_stats, accuracy, shift = evaluate(
          step,
          model,
          feedback,
          extras={'loss': cur_loss},
          verbose=FLAGS.verbose_logging,
      )
      wandb.log({'train loss':train_stats['loss'], 'train match':train_stats['match'],'s_g_average':accuracy, 'average_shift':shift}, )
      logging.info('(train) step %d: %s', step, {'average_accuracy':accuracy, 'average_shift':shift})

      # Validation info.
      val_feedback = val_sampler.next()  # full-batch
      val_stats, accuracy, shift = evaluate(
          step, model, val_feedback, verbose=FLAGS.verbose_logging)
      wandb.log({'validation match':val_stats['match'],'s_g_average_val':accuracy, 'average_shift_val': shift})
      logging.info('(val) step %d: %s', step, {'average_accuracy':accuracy, 'average_shift':shift,})

      # If best scores, update checkpoint.

      def checkpoint_model_custom(model, best_average_accuracy, best_average_shift, average_accuracy, average_shift):
        if average_accuracy > best_average_accuracy:
          logging.info('Saving new checkpoint for average accuracy...')
          best_average_accuracy = average_accuracy
          model.save_model('best_average_accuracy.pkl')
        
        if average_shift < best_average_shift:
          logging.info('Saving new checkpoint for average shift...')
          best_average_shift = average_shift
          model.save_model('best_average_shift.pkl')
        return best_average_accuracy, best_average_shift

      def checkpoint_model_default(model, score):
        if score > best_score:
          logging.info('Saving new checkpoint...')
          best_score = score
          model.save_model('best.pkl')
          return best_score

      score = val_stats['score']
      #best_score = checkpoint_model_default(model, score)

      best_average_accuracy, best_average_shift = checkpoint_model_custom(
        model,
        best_average_accuracy, 
        best_average_shift,
        accuracy,
        shift
      )

      # if score > best_score:
      #   logging.info('Saving new checkpoint...')
      #   best_score = score
      #   model.save_model('best.pkl')

  # Training complete, evaluate on test set.
  def test_model_custom(model):
    test_sampler, _ = clrs.clrs21_test(FLAGS.algorithm)
    logging.info('Restoring best average accuracy model from checkpoint...')
    model.restore_model('best_average_accuracy.pkl', only_load_processor=False)

    test_feedback = test_sampler.next()  # full-batch
    test_stats, a, s = evaluate(
        step, model, test_feedback, verbose=FLAGS.verbose_logging)
    logging.info('(test) step %d: %s', step, {'average_accuracy':a, 'average_shift':s,})
    wandb.log({'test_accuracy': test_stats['match'], 'average accuracy': a, 'average shift':s,})


    logging.info('Restoring best average shift model from checkpoint...')
    model.restore_model('best_average_shift.pkl', only_load_processor=False)
    test_feedback = test_sampler.next()  # full-batch
    test_stats, a, s = evaluate(
        step, model, test_feedback, verbose=FLAGS.verbose_logging)
    logging.info('(test) step %d: %s', step, {'average_accuracy':a, 'average_shift':s,})
    wandb.log({'test_accuracy': test_stats['match'], 'average accuracy': a, 'average shift':s,})

    
  def test_model_default(model):
    test_sampler, _ = clrs.clrs21_test(FLAGS.algorithm)
    logging.info('Restoring best model from checkpoint...')
    model.restore_model('best.pkl', only_load_processor=False)

    test_feedback = test_sampler.next()  # full-batch
    test_stats, a, s = evaluate(
        step, model, test_feedback, verbose=FLAGS.verbose_logging)
    logging.info('(test) step %d: %s', step, test_stats)
    wandb.log({'test_accuracy': test_stats['match'], 'average accuracy': a, 'average shift':s,})
  
  test_model_custom(model)


    
  # test_sampler, _ = clrs.clrs21_test(FLAGS.algorithm)
  # logging.info('Restoring best model from checkpoint...')
  # model.restore_model('best.pkl', only_load_processor=False)

  # test_feedback = test_sampler.next()  # full-batch
  # test_stats, a, s = evaluate(
  #     step, model, test_feedback, verbose=FLAGS.verbose_logging)
  # logging.info('(test) step %d: %s', step, test_stats)
  # wandb.log({'test_accuracy': test_stats['match'], 'average accuracy': a, 'average shift':s,})


if __name__ == '__main__':
  app.run(main)
