
"""Generic training script that trains a model using a given dataset."""
import os
import os.path as ops
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils import tf_utils,log_utils
import numpy as np
import time
import cv2

from tensorflow.python import debug as tfdbg
#import ipdb

from tensorflow.core.protobuf import saver_pb2

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC' 
IMAGE_H = 64
IMAGE_W = 512 
SEQ_LENGTH = 64*np.ones(32) 
#SEQ_LENGTH = 15*np.ones(32) #batch_size * (IMAGE_W/filter[2*2] -1)
MAX_CHAR_LEN = 11
_FILE_PATTERN = 'ocr_%s_*.tfrecord'

logger = log_utils.init_logger()

# =========================================================================== #
# General Flags.
# need modify parameters:
# - dataset_name
# - num_classes
# - model_name
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.85, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_boolean('sync_replicas', False,
                            'Whether or not to synchronize the replicas during training.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'ocr', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 10, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'crnn', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')


# =========================================================================== #
# CRNN TRAIN Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'LR_DECAY_STEPS', 10000, 'Set the learning rate decay steps.')
tf.app.flags.DEFINE_float(
    'LR_DECAY_RATE', 0.1, 'Set the learning rate decay rate.')
tf.app.flags.DEFINE_float(
    'LEARNING_RATE', 0.1, 'Set the initial learning rate.')
tf.app.flags.DEFINE_boolean(
    'TF_ALLOW_GROWTH', True,
    'Set the GPU allow growth parameter during tensorflow training process.')
tf.app.flags.DEFINE_integer(
    'EPOCHS', 1000, 'Set the shadownet training epochs.')
tf.app.flags.DEFINE_integer(
    'DISPLAY_STEP', 1, 'Set the display step.')


FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
        
    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():

        # =================================================================== #
        # Config model_deploy.                                                #
        # Keep TF Slim Models structure.                                      #
        # Useful if want to need multiple GPUs and/or servers in the future.  #
        # =================================================================== #
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)
        
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # =================================================================== #
        # Select the dataset.
        # =================================================================== #
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        
        # =================================================================== #
        # Select the network
        # =================================================================== #
        if FLAGS.model_name == "crnn":
            crnn_net = nets_factory.get_network(FLAGS.model_name)
            network_fn = crnn_net(phase='Train', 
                                  num_classes=(dataset.num_classes - FLAGS.labels_offset))

        else:
            network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=(dataset.num_classes - FLAGS.labels_offset),
                    weight_decay=FLAGS.weight_decay,
                    is_training=True)
            
        # =================================================================== #
        # Select the preprocessing function.
        # =================================================================== #
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                                                        preprocessing_name, 
                                                        is_training=True)

        #tf_utils.print_configuration(FLAGS.__flags,
        #                             dataset.data_sources, save_dir=FLAGS.train_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #        
        with tf.device(deploy_config.inputs_device()):
            if FLAGS.dataset_name == "ocr":
                image,label = tf_utils.read_features(ops.join(FLAGS.dataset_dir, 
                                                              "ocr_train_000.tfrecord"),
                                                                num_epochs=None)
                
            else:           
                with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                    provider = slim.dataset_data_provider.DatasetDataProvider(
                            dataset,
                            num_readers=FLAGS.num_readers,
                            common_queue_capacity=20 * FLAGS.batch_size,
                            common_queue_min=10 * FLAGS.batch_size,
                            shuffle=True)
            
                [image, label] = provider.get(['image', 'label'])

            # Pre-processing image, labels and bboxes.
            train_image_size = FLAGS.train_image_size or network_fn.default_image_size
            
            #image = image_preprocessing_fn(image, 32, 256)
                
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [IMAGE_H, IMAGE_W],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)

            #label = tf.reshape(label,[MAX_CHAR_LEN])
            images, labels = tf.train.shuffle_batch(
                    tensors=[image, label], 
                    batch_size=32, 
                    capacity=1000+2*32, 
                    min_after_dequeue=100, 
                    #enqueue_many=True,
                    num_threads=1)            
            images = tf.cast(x=images, dtype=tf.float32)

            
            if FLAGS.model_name != "crnn":
              labels = slim.one_hot_encoding(
                  labels, dataset.num_classes - FLAGS.labels_offset)
              batch_queue = slim.prefetch_queue.prefetch_queue(
                  [images, labels], capacity=2 * deploy_config.num_clones)
                       
        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        #def clone_fn(batch_queue):
        def clone_fn(images,labels):
            """Allows data parallelism by creating multiple
            clones of network_fn."""
            # Dequeue batch.
            #images, labels = batch_queue.dequeue()
            with tf.variable_scope('crnn'):
                logits, end_points = network_fn.build_CRNNnet(images)
                           
            #############################
            # Specify the loss function #
            #############################

            if FLAGS.model_name == "crnn":
                if FLAGS.dataset_name == "mnist":
                    idx = tf.where(tf.not_equal(labels,0))
                    labels = tf.SparseTensor(idx,tf.gather_nd(labels,idx),labels.get_shape())
                    labels = tf.cast(labels, tf.int32)
               
                ctc_loss = tf.nn.ctc_loss(labels=labels, 
                                          inputs=logits, 
                                          sequence_length=SEQ_LENGTH,
                                          ctc_merge_repeated=True,
                                          ignore_longer_outputs_than_inputs=True,
                                          time_major=True)

                ctc_loss = tf.reduce_mean(ctc_loss)
                ctc_loss = tf.Print(ctc_loss, [ctc_loss], message='* Loss : ')
                
                tf.losses.add_loss(ctc_loss)
                decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, sequence_length=SEQ_LENGTH, merge_repeated=False)
                
                sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

            else:
                if 'AuxLogits' in end_points:
                    slim.losses.softmax_cross_entropy(
                        end_points['AuxLogits'], labels,
                        label_smoothing=FLAGS.label_smoothing, weights=0.4,
                        scope='aux_loss')
                slim.losses.softmax_cross_entropy(
                  logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
            
            return end_points,ctc_loss,sequence_dist,labels,decoded

        if  FLAGS.model_name == "crnn":
            end_points,ctc_loss,sequence_dist,labels,decoded = clone_fn(images,labels)
            
            network_fn.train_crnn(FLAGS,global_step,ctc_loss,sequence_dist,labels,decoded)
            
        else:
            # =================================================================== #
            # Add summaries from first clone.
            # =================================================================== #   
            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
                 
            #clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
            clones = model_deploy.create_clones(deploy_config, clone_fn, [images,labels])
            first_clone_scope = deploy_config.clone_scope(0)
            # Gather update_ops from the first clone. These contain, for example,
            # the updates for the batch_norm variables created by network_fn.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    
            # Add summaries for end_points.
            end_points,ctc_loss,sequence_dist,labels,decoded = clones[0].outputs
                    
        
            for end_point in end_points:
                x = end_points[end_point]
                summaries.add(tf.summary.histogram('activations/' + end_point, x))
                summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                                tf.nn.zero_fraction(x)))
            # Add summaries for losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
                summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
            summaries.add(tf.summary.scalar('losses/ctc_loss', tensor=ctc_loss))
            summaries.add(tf.summary.scalar('Seq_Dist', tensor=sequence_dist))
                
            # Add summaries for variables.
            for variable in slim.get_model_variables():
                summaries.add(tf.summary.histogram(variable.op.name, variable))
    
    
            # =================================================================== #
            # Configure the moving averages.
            # =================================================================== #
            if FLAGS.moving_average_decay:
                moving_average_variables = slim.get_model_variables()
                variable_averages = tf.train.ExponentialMovingAverage(
                    FLAGS.moving_average_decay, global_step)
            else:
                moving_average_variables, variable_averages = None, None
    
            # =================================================================== #
            # Configure the optimization procedure.
            # =================================================================== #
            with tf.device(deploy_config.optimizer_device()):
                learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                                 dataset.num_samples,
                                                                 global_step)
                #optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate).minimize(cost)
                optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
                #optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate).minimize(loss=ctc_loss, global_step=global_step)
                summaries.add(tf.summary.scalar('learning_rate', tensor=learning_rate))
    
            if FLAGS.sync_replicas:
              # If sync_replicas is enabled, the averaging will be done in the chief
              # queue runner.
              optimizer = tf.train.SyncReplicasOptimizer(
                  opt=optimizer,
                  replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                  total_num_replicas=FLAGS.worker_replicas,
                  variable_averages=variable_averages,
                  variables_to_average=moving_average_variables)
            elif FLAGS.moving_average_decay:
                # Update ops executed locally by trainer.
                update_ops.append(variable_averages.apply(moving_average_variables))
    
            # Variables to train.
            variables_to_train = tf_utils.get_variables_to_train(FLAGS)
    
            # and returns a train_tensor and summary_op
            total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                #regularization_losses = ctc_loss,
                var_list=variables_to_train)
            
            # Add total_loss to summary.
            summaries.add(tf.summary.scalar('total_loss', total_loss))
    
            # Create gradient updates.
            grad_updates = optimizer.apply_gradients(clones_gradients,
                                                     global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')
                train_tensor = slim.learning.create_train_op(total_loss, optimizer)
                
            
            """    
            train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                              name='train_op')
            """
    
            # Add the summaries from the first clone. These contain the summaries
            #summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
            #                                   first_clone_scope))
            # Merge all summaries together.
            summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
            # =================================================================== #
            # Configure the saver procedure.
            # =================================================================== #        
            saver = tf.train.Saver(max_to_keep=5,
                                   keep_checkpoint_every_n_hours=1.0,
                                   write_version=2,
                                   pad_step_number=False)
            
            model_save_dir = './checkpoints/' + FLAGS.model_name
            if not ops.exists(model_save_dir):
                os.makedirs(model_save_dir)
            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            model_name = 'CRNNnet_{:s}.ckpt'.format(str(train_start_time))
            model_save_path = ops.join(model_save_dir, model_name)
            
            # =================================================================== #
            # Kicks off the training.
            # =================================================================== #
            #summary_writer = tf.summary.FileWriter("tensorboard_%d" %(ctx.worker_num),graph=tf.get_default_graph())
            
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
            config = tf.ConfigProto(log_device_placement=False,
                                    gpu_options=gpu_options)
    
            slim.learning.train(
                train_tensor,
                logdir=FLAGS.train_dir,
                master='',
                is_chief=True,
                init_fn=tf_utils.get_init_fn(FLAGS),
                summary_op=summary_op,
                number_of_steps=FLAGS.max_number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                saver=saver,
                save_interval_secs=FLAGS.save_interval_secs,
                session_config=config,
                #session_wrapper=tfdbg.LocalCLIDebugWrapperSession, 
                sync_optimizer=None)

if __name__ == '__main__':
    tf.app.run()
