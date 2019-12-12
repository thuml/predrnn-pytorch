import numpy as np
import random

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.load()

    def load(self):
        dat_1 = np.load(self.paths[0])
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        if self.num_paths == 2:
            dat_2 = np.load(self.paths[1])
            num_clips_1 = dat_1['clips'].shape[1]
            dat_2['clips'][:,:,0] += num_clips_1
            self.data['clips'] = np.concatenate(
                (dat_1['clips'], dat_2['clips']), axis=1)
            self.data['input_raw_data'] = np.concatenate(
                (dat_1['input_raw_data'], dat_2['input_raw_data']), axis=0)
            self.data['output_raw_data'] = np.concatenate(
                (dat_1['output_raw_data'], dat_2['output_raw_data']), axis=0)
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def total(self):
        return self.data['clips'].shape[1]

    def begin(self, do_shuffle = True):
        self.indices = np.arange(self.total(),dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind
                                        in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind
                                         in self.current_batch_indices)

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind
                                        in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind
                                         in self.current_batch_indices)

    def no_batch_left(self):
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False

    def input_batch(self):
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.current_batch_size, self.current_input_length) +
            tuple(self.data['dims'][0])).astype(self.input_data_type)
        input_batch = np.transpose(input_batch,(0,1,3,4,2))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][0, batch_ind, 0]
            end = self.data['clips'][0, batch_ind, 0] + \
                    self.data['clips'][0, batch_ind, 1]
            data_slice = self.data['input_raw_data'][begin:end, :, :, :]
            data_slice = np.transpose(data_slice,(0,2,3,1))
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def output_batch(self):
        if self.no_batch_left():
            return None
        if(2 ,3) == self.data['dims'].shape:
            raw_dat = self.data['output_raw_data']
        else:
            raw_dat = self.data['input_raw_data']
        if self.is_output_sequence:
            if (1, 3) == self.data['dims'].shape:
                output_dim = self.data['dims'][0]
            else:
                output_dim = self.data['dims'][1]
            output_batch = np.zeros(
                (self.current_batch_size,self.current_output_length) +
                tuple(output_dim))
        else:
            output_batch = np.zeros((self.current_batch_size, ) +
                                    tuple(self.data['dims'][1]))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][1, batch_ind, 0]
            end = self.data['clips'][1, batch_ind, 0] + \
                    self.data['clips'][1, batch_ind, 1]
            if self.is_output_sequence:
                data_slice = raw_dat[begin:end, :, :, :]
                output_batch[i, : data_slice.shape[0], :, :, :] = data_slice
            else:
                data_slice = raw_dat[begin, :, :, :]
                output_batch[i,:, :, :] = data_slice
        output_batch = output_batch.astype(self.output_data_type)
        output_batch = np.transpose(output_batch, [0,1,3,4,2])
        return output_batch

    def get_batch(self):
        input_seq = self.input_batch()
        output_seq = self.output_batch()
        batch = np.concatenate((input_seq, output_seq), axis=1)
        return batch
