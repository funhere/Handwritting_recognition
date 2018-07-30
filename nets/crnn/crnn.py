#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Refer from: https://github.com/MaybeShewill-CV/CRNN_Tensorflow/blob/master/crnn_model/crnn_model.py
# @Site    : http://github.com/TJCVRS

"""
Implement the crnn model mentioned in An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition paper
"""
import os
import os.path as ops
import time

import numpy as np
from collections import namedtuple
from utils import tf_utils,log_utils

import tensorflow as tf
#from utils import tf_extended

from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn


from nets.crnn import cnn_basenet
logger = log_utils.init_logger()

HIDDEN_NUMS = 256
NUM_CLASSES = 10

# Test setting
is_recursive = False # Recursively test the dataset 
loops_nums = 100

class CRNNnet(cnn_basenet.CNNBaseModel):
    """
        Implement the crnn model for squence recognition
    """
    #def __init__(self, phase, hidden_nums, layers_nums, seq_length, num_classes):
    def __init__(self, phase, num_classes=NUM_CLASSES+1):
        """

        :param phase:
        """
        super(CRNNnet, self).__init__()
        self.__phase = phase
        #self.__hidden_nums = hidden_nums
        #self.__layers_nums = layers_nums
        #self.__seq_length = seq_length
        self.__num_classes = num_classes
        return

    @property
    def phase(self):
        """

        :return:
        """
        return self.__phase

    @phase.setter
    def phase(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, str):
            raise TypeError('value should be a str \'Test\' or \'Train\'')
        if value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()
        return

    def __conv_stage(self, inputdata, out_dims, name=None):
        """
        Traditional conv stage in VGG format
        :param inputdata:
        :param out_dims:
        :return:
        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name)
        relu = self.relu(inputdata=conv)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def __feature_sequence_extraction(self, inputdata):
        """
        Implement the 2.1 Part Feature Sequence Extraction
        :param inputdata: eg. batch*32*128*3 NHWC format
        :return:
            end_points: a set of activations for external use, for example summaries or
                losses.
        """
        # end_points collect relevant activations for external use.
        end_points = {}
  
        conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*64*64
        end_points['conv1'] = conv1
        conv2 = self.__conv_stage(inputdata=conv1, out_dims=128, name='conv2')  # batch*8*32*128
        end_points['conv2'] = conv2
        conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3')  # batch*8*32*256        
        relu3 = self.relu(conv3) # batch*8*32*256 
        end_points['conv3'] = relu3
        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4')  # batch*8*32*256
        relu4 = self.relu(conv4)  # batch*8*32*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID')  # batch*4*32*256
        end_points['conv4'] = max_pool4
        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5')  # batch*4*32*512
        relu5 = self.relu(conv5)  # batch*4*32*512
        if self.phase.lower() == 'train':
            bn5 = self.layerbn(inputdata=relu5, is_training=True)
        else:
            bn5 = self.layerbn(inputdata=relu5, is_training=False)  # batch*4*32*512
        max_pool5 = self.maxpooling(inputdata=bn5, kernel_size=2, stride=2) #maxpool H and W
        end_points['conv5'] = max_pool5
        conv6 = self.conv2d(inputdata=max_pool5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6')  # batch*4*32*512
        relu6 = self.relu(conv6)  # batch*4*32*512
        if self.phase.lower() == 'train':
            bn6 = self.layerbn(inputdata=relu6, is_training=True)
        else:
            bn6 = self.layerbn(inputdata=relu6, is_training=False)  # batch*4*32*512
        max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[2, 1], stride=[2, 1])  # batch*2*32*512
        end_points['conv6'] = max_pool6
        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7')  # batch*1*32*512
        relu7 = self.relu(conv7)  # batch*1*32*512
        end_points['conv7'] = relu7
        return relu7, end_points

    def __map_to_sequence(self, inputdata):
        """
        Implement the map to sequence part of the network mainly used to convert the cnn feature map to sequence used in
        later stacked lstm layers
        :param inputdata:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return self.squeeze(inputdata=inputdata, axis=1)

    def __sequence_label(self, inputdata):
        """
        Implement the sequence label part of the network
        :param inputdata:
        :return:
        """
        list_n_hidden = [HIDDEN_NUMS, HIDDEN_NUMS]
        
        with tf.variable_scope('LSTMLayers'):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                         dtype=tf.float32)

            if self.phase.lower() == 'train':
                stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer, keep_prob=0.5)

            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])  # [batch x width, 2*n_hidden]

            w = tf.Variable(tf.truncated_normal([hidden_nums, self.__num_classes], stddev=0.1), name="w")
            # Doing the affine projection

            logits = tf.matmul(rnn_reshaped, w)

            logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred

    def build_CRNNnet(self, inputdata):
        """

        :param inputdata:
        :return:
            net_out：output tensor corresponding to the final_endpoint.
            end_points: a set of activations for external use, for example summaries or
                losses.
        """
        # first apply the cnn feature extraction stage
        cnn_out,end_points = self.__feature_sequence_extraction(inputdata=inputdata)
        print("====##===cnn_out:%s",cnn_out.get_shape().as_list() )
        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(inputdata=cnn_out)
        print("====##===sequence:%s",sequence.get_shape().as_list() )
        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(inputdata=sequence)
        print("====net_out===:",net_out.get_shape().as_list() )
        print("====Predictions===:",raw_pred.get_shape().as_list() )
        end_points['Logits'] = net_out
        end_points['Predictions'] = raw_pred

        return net_out, end_points


    def train_crnn(self, FLAGS,global_step,cost,sequence_dist,input_labels,pred_labels):
        """
        Train crnn model, collect summaries, save model.
        :param inputdata:
            FLAGS: config parameters
            global_step: global step of optimizer
            cost: loss cost
            sequence_dist:
            input_labels: ground true labels
            pred_labels: predict labels
        :return:
        """
        learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step,
                                                   FLAGS.LR_DECAY_STEPS, FLAGS.LR_DECAY_RATE,
                                                   staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)
    
        # Set tf summary
        tboard_save_path = 'tboard/crnn'
        if not ops.exists(tboard_save_path):
            os.makedirs(tboard_save_path)
        tf.summary.scalar(name='Cost', tensor=cost)
        tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
        tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
        merge_summary_op = tf.summary.merge_all()
    
        # Set the training parameters
        train_epochs = FLAGS.EPOCHS
        checkpoint_path =  FLAGS.checkpoint_path
    
        # Set saver configuration
        #saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
        saver = tf.train.Saver()
        model_save_dir = 'checkpoints/crnn'
        #model_save_dir = FLAGS.checkpoint_path
        if not ops.exists(model_save_dir):
            os.makedirs(model_save_dir)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'crnn_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = ops.join(model_save_dir, model_name)
    
        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
        sess_config.gpu_options.allow_growth = FLAGS.TF_ALLOW_GROWTH
    
        sess = tf.Session(config=sess_config)
    
        summary_writer = tf.summary.FileWriter(tboard_save_path)
        summary_writer.add_graph(sess.graph)
    
        with sess.as_default():
            if checkpoint_path is None:
                logger.info('Training from scratch')
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                logger.info('Restore model from {:s}'.format(checkpoint_path))
                saver.restore(sess=sess, save_path=checkpoint_path)
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
            for epoch in range(train_epochs):
                _, c, seq_distance, preds, gt_labels, summary = sess.run(
                    [optimizer, cost, sequence_dist, pred_labels, input_labels, merge_summary_op])
    
                # calculate the precision 
                preds = tf_utils.sparse_tensor_to_str(preds[0])
                gt_labels = tf_utils.sparse_tensor_to_str(gt_labels)
        
                accuracy = []    
    
                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            #import ipdb; ipdb.set_trace()
                            #print("tmp,pred:",tmp, pred[i])
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / totol_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                #
                if epoch % FLAGS.DISPLAY_STEP == 0:
                    logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, c, seq_distance, accuracy))
    
                summary_writer.add_summary(summary=summary, global_step=epoch)
                #logger.info('Save model to {:s}'.format(model_save_path))
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                #saver.save(sess,"/tmp/crnn.ckpt")
    
            coord.request_stop()
            coord.join(threads=threads)
       
        sess.close()
         
        return


    def eval_crnn(self, FLAGS, decoded, images_sh, labels_sh):
        """
        Evaluation crnn model, collect summaries, save model.
        :param inputdata:
            FLAGS: config parameters
            decoded: predict labels
            images_sh: image data
            labels_sh: ground true labels
        :return:
        """   
        # config tf session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
        sess_config.gpu_options.allow_growth = FLAGS.TF_ALLOW_GROWTH
        
        # config tf saver
        #saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()
        
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
    
        sess = tf.Session(config=sess_config)
        #checkpoint_path = "checkpoints/crnn/crnn_2018-07-20-12-09-01.ckpt-99"
        #module_file =  tf.train.latest_checkpoint('/Users/simon/Desktop/OCR/Handwritting_recognition/checkpoints/crnn')
        """
        test_sample_count = 0
        for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir, 'test_feature.tfrecords')):
            test_sample_count += 1
        loops_nums = int(math.ceil(test_sample_count / 32))
        # loops_nums = 100
        """
    
        with sess.as_default():
    
            # restore the model weights
            saver.restore(sess=sess, save_path=checkpoint_path)
            #saver.restore(sess,"/tmp/crnn.ckpt")
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
            print('Start predicting ......')
            #if not is_recursive:
            if not is_recursive:
                predictions, images, labels = sess.run([decoded, images_sh, labels_sh])
                
                preds_res = tf_utils.sparse_tensor_to_str(predictions[0])
                gt_res = tf_utils.sparse_tensor_to_str(labels)
                #preds_res = tf_utils.decode_sparse_tensor(predictions[0])
                #gt_res = tf_utils.decode_sparse_tensor(labels)
                print("$$$$$preds:",preds_res)
                print("$$$$$gt_labels:",gt_res)
                accuracy = []
                #true_numer = 0
    
                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
                    total_count = len(gt_label)

                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / total_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)

                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                logger.info('Mean test accuracy is {:5f}'.format(accuracy))
                print('Mean test accuracy is {:5f}'.format(accuracy))
                
                #for index, image in enumerate(images):
                    #print('Predict image with gt label: {:s} **** predict label: {:s}'.format(
                    #    gt_res[index], preds_res[index]))
                    #if is_vis:
                    #    plt.imshow(image[:, :, (2, 1, 0)])
                    #    plt.show()
            else:
                accuracy = []
                for epoch in range(loops_nums):
                    predictions, images, labels = sess.run([decoded, images_sh, labels_sh])

                    preds_res = tf_utils.sparse_tensor_to_str(predictions[0])
                    gt_res = tf_utils.sparse_tensor_to_str(labels)
    
                    for index, gt_label in enumerate(gt_res):
                        pred = preds_res[index]
                        totol_count = len(gt_label)
                        correct_count = 0
                        try:
                            for i, tmp in enumerate(gt_label):
                                if tmp == pred[i]:
                                    correct_count += 1
                        except IndexError:
                            continue
                        finally:
                            try:
                                accuracy.append(correct_count / totol_count)
                            except ZeroDivisionError:
                                if len(pred) == 0:
                                    accuracy.append(1)
                                else:
                                    accuracy.append(0)
    
                    for index, image in enumerate(images):
                        print('Predict image with gt label: {:s} **** predict label: {:s}'.format(
                            gt_res[index], preds_res[index]))
                        
                        # if is_vis:
                        #     plt.imshow(image[:, :, (2, 1, 0)])
                        #     plt.show()
    
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
                print('Test accuracy is {:5f}'.format(accuracy))
           
                
            coord.request_stop()
            coord.join(threads=threads)
              
        sess.close()
        
        return


# Temporary setting for OCR Telnumber 
CRNNnet.default_image_size = (32,256)
#CRNNnet.default_image_size = (64,512)

    
def crnn_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the arg scope.
    Args:
      weight_decay: The l2 regularization coefficient.
    Returns:
      An arg_scope.
    """
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc
     """       