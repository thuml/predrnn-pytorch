__author__ = 'jianjin'

import numpy as np
import os
from PIL import Image
import tensorflow as tf
import logging
import random

logger = logging.getLogger(__name__)


class InputHandle:
    def __init__(self, datas, indices, configs):
        self.name = configs['name'] + ' iterator'
        self.minibatch_size = configs['batch_size']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = configs['seq_length']
        self.injection_action = configs['injection_action']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        else:
            self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_height, self.image_width, 7)).astype(np.float32)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind[-1]
            end = begin + self.current_input_length
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(batch_ind[0]):
                if k == batch_ind[1]:
                    example = tf.train.Example()
                    example.ParseFromString(serialized_example)
                    break
                k += 1
            for j in range(begin, end):
                action_name = str(j) + '/action'
                action_value = np.array(example.features.feature[action_name].float_list.value)
                if action_value.shape == (0,):  # End of frames/data
                    print("error! " + str(batch_ind))
                input_batch[i, j - begin, :, :, 3:] = np.stack([np.ones([64, 64]) * i for i in action_value], axis=2)

                # endeffector_pos_name = str(j) + '/endeffector_pos'
                # endeffector_pos_value = list(example.features.feature[endeffector_pos_name].float_list.value)
                # endeffector_positions = np.vstack((endeffector_positions, endeffector_pos_value))

                aux1_image_name = str(j) + '/image_aux1/encoded'
                aux1_byte_str = example.features.feature[aux1_image_name].bytes_list.value[0]
                aux1_img = Image.frombytes('RGB', (64, 64), aux1_byte_str)
                aux1_arr = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))

                # main_image_name = str(j) + '/image_main/encoded'
                # main_byte_str = example.features.feature[main_image_name].bytes_list.value[0]
                # main_img = Image.frombytes('RGB', (64, 64), main_byte_str)
                # main_arr = np.array(main_img.getdata()).reshape((main_img.size[1], main_img.size[0], 3))

                input_batch[i, j - begin, :, :, :3] = aux1_arr.reshape(64, 64, 3) / 255
        input_batch = input_batch.astype(np.float32)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))


class DataProcess:
    def __init__(self, configs):
        self.configs = configs
        self.train_data_path = configs['train_data_paths']
        self.valid_data_path = configs['valid_data_paths']
        self.image_height = configs['image_height']
        self.image_width = configs['image_width']
        self.seq_len = configs['seq_length']

    def load_data(self, path, mode='train'):
        path = os.path.join(path[0], 'softmotion30_44k')
        if mode == 'train':
            path = os.path.join(path, 'train')
        elif mode == 'test':
            path = os.path.join(path, 'test')
        else:
            print("ERROR!")
        print('begin load data' + str(path))

        video_fullpaths = []
        indices = []

        tfrecords = os.listdir(path)
        tfrecords.sort()
        num_pictures = 0

        for tfrecord in tfrecords:
            filepath = os.path.join(path, tfrecord)
            video_fullpaths.append(filepath)
            k = 0
            for serialized_example in tf.compat.v1.python_io.tf_record_iterator(os.path.join(path, tfrecord)):
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                i = 0
                while True:
                    action_name = str(i) + '/action'
                    action_value = np.array(example.features.feature[action_name].float_list.value)
                    if action_value.shape == (0,):  # End of frames/data
                        break
                    i += 1
                num_pictures += i
                for j in range(i - self.seq_len + 1):
                    indices.append((filepath, k, j))
                k += 1
        print("there are " + str(num_pictures) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return video_fullpaths, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.train_data_path, mode='train')
        return InputHandle(train_data, train_indices, self.configs)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.valid_data_path, mode='test')
        return InputHandle(test_data, test_indices, self.configs)
