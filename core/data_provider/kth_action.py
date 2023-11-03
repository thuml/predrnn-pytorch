__author__ = 'gaozhifeng'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from typing import Iterable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

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
        # Create batch of N videos of length L, i.e. shape = (N, L, w, h, c)
        # where w x h is the resolution and c the number of color channels
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 1)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


@dataclass
class ActionFrameInfo:
    file_name: str
    file_path: str
    person_mark: int
    category_flag: int


class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']  # path to parent folder containing category dirs
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.categories = self.category_1 + self.category_2
        self.image_width = input_param['image_width']

        # Hard coded training and test persons (prevent same person occurring in train - test set)
        self.train_person = ['01', '02', '03', '04', '05', '06', '07', '08',
                             '09', '10', '11', '12', '13', '14', '15', '16']
        self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']


    def generate_frames(self, root_path, person_ids: List[int]) -> Iterable[ActionFrameInfo]:
        """Generate frame info for all frames.
        
        Parameters:
            person_ids: persons to include
        """
        person_mark = 0
        for cat_dir in self.categories: # handwaving
            if cat_dir in self.category_1:
                frame_category_flag = 1 # 20 step
            elif cat_dir in self.category_2:
                frame_category_flag = 2 # 3 step
            else:
                raise Exception("category error!!!")

            cat_dir_path = os.path.join(root_path, cat_dir)
            cat_videos = os.listdir(cat_dir_path)

            for person_direction_video in cat_videos:
                person_id = person_direction_video[6:8]  # chars 6-8 contan number
                if person_id not in person_ids:
                    continue
                person_mark += 1  # identify all stored frames as belonging to this person + direction
                dir_path = os.path.join(cat_dir_path, person_direction_video)
                filelist = os.listdir(dir_path)
                filelist.sort() 
                for frame_name in filelist: 
                    if frame_name.startswith('image') == False:
                        continue
                    yield ActionFrameInfo(
                        file_name=frame_name,
                        file_path=os.path.join(dir_path, frame_name),
                        person_mark=person_mark,
                        category_flag=frame_category_flag
                    )

    def load_data(self, paths, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        if mode == 'train':
            mode_person_ids = self.train_person
        elif mode == 'test':
            mode_person_ids = self.test_person
        else:
            raise Exception("Unexpected mode: " + mode)
        print('begin load data' + str(path))

        frames_file_name = []
        frames_person_mark = []  # for each frame in the joint array, mark the person ID
        frames_category = []

        # First count total number of frames
        # Do it without creating massive array:
        # all_frame_info = self.generate_frames(path, mode_person_ids)
        tot_num_frames = sum((1 for _ in self.generate_frames(path, mode_person_ids)))
        print(f"Preparing to load {tot_num_frames} video frames.")
        
        # Target array containing ALL RESIZED frames
        data = np.empty((tot_num_frames, self.image_width, self.image_width , 1),
                        dtype=np.float32)  # np.float32

        # Read, resize, and store video frames
        for i, frame in enumerate(self.generate_frames(path, mode_person_ids)):
            frame_im = Image.open(frame.file_path).convert('L') # int8 2D array

            # input type must be float32 for default interpolation method cv2.INTER_AREA
            frame_np = np.array(frame_im, dtype=np.uint8)  # (1000, 1000) numpy array
            data[i,:,:,0] = (cv2.resize(
                frame_np, (self.image_width,self.image_width))/255.0).astype(np.float32)

            frames_file_name.append(frame.file_name)
            frames_person_mark.append(frame.person_mark)
            frames_category.append(frame.category_flag)
            
        # identify sequences of <seq_len> within the same video
        indices = []
        seq_end_idx = len(frames_person_mark) - 1
        while seq_end_idx >= self.seq_len - 1:
            seq_start_idx = seq_end_idx - self.seq_len + 1
            if frames_person_mark[seq_end_idx] == frames_person_mark[seq_start_idx]:
                # Get person ID at the start and end of this sequence (of seq_len)
                end = int(frames_file_name[seq_end_idx][6:10])
                start = int(frames_file_name[seq_start_idx][6:10])
                
                # TODO: mode == 'test'
                if end - start == self.seq_len - 1:
                    # Save index into OUT data array indicating start point of sequence
                    indices.append(seq_start_idx)

                    # The step size depends on the category
                    if frames_category[seq_end_idx] == 1:
                        seq_end_idx -= self.seq_len - 1
                    elif frames_category[seq_end_idx] == 2:
                        seq_end_idx -= 2
                    else:
                        raise Exception("category error 2 !!!")
            
            seq_end_idx -= 1

        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

