# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :18-6-26 下午12:36
# File     :cnnNetwork.py
# Location:/Home/PycharmProjects/..


# Acquire hand-writing data
# 28*28 picture object
# For each tag is 0-9
# one-hot code to 10 dimensions vector
import numpy as np


# Loading class
# extend to ImageLoader and LabelLoader
class Loader(object):

    # Construct
    # path: file path
    # count: sample count
    def __init__(self, path, count):
        self.path = path
        self.count = count

    # Function: read file and return context
    def get_file_context(self):
        print(self.path)
        f = open(self.path, 'rb')
        # read byte stream
        context = f.read()
        f.close()
        # return byte array
        return context

    # Trans the unsigned byte to int
    # def to_int(self, byte):
    #     return Struct.unpack('B', byte)[0]


# ImageLoader
class ImageLoader(Loader):

    # Function: Acquire the index's data from byte array
    # In byte array contains all pic data
    @staticmethod
    def get_picture(context, index):
        # file header is 16 byte
        # 28*28 byte for one pic
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            # add one px
            picture.append([])
            for j in range(28):
                byte1 = context[start + i * 28 + j]
                picture[i].append(byte1)
                # add one px for each row
                # picture[i].append(self.to_int(byte1))
        # pic is the list like [[x,x,x..][x,x,x...][x,x,x...][x,x,x...]]
        return picture

    # Trans the pic to the 784 ROW VECTOR patten
    @staticmethod
    def get_one_sample(picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    # Load data
    # Acquire the all sample input vector
    # one_row represent if Trans to ROW VECTOR
    def load(self, one_row=False):
        # Acquire the byte array of context
        context = self.get_file_context()
        data_set = []
        # Iteration for each sample
        for index in range(self.count):
            # Acquire the index's sample in data collection
            # return 2 dimensions array
            inn_pic = self.get_picture(context, index)
            if one_row:
                # Trans to 1 dimension patten
                inn_pic = self.get_one_sample(inn_pic)
            data_set.append(inn_pic)
        return data_set


# LabelLoader
class LabelLoader(Loader):
    # Load the file
    # Acquire All samples label vectors
    def load(self):
        # Acquire byte array
        context = self.get_file_context()
        labels = []
        # Iteration for each sample
        for index in range(self.count):
            # file header has 8 bytes
            one_label = context[index + 8]
            # one-hot code
            one_label_vec = self.norm(one_label)
            labels.append(one_label_vec)
        return labels

    # one-hot code
    # Trans a value to 10 dimensions label vector
    @staticmethod
    def norm(label):
        label_vec = []
        # label_value = self.to_int(label)
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec


# Acquire trained collection
# one_row represent if Trans to ROW VECTOR
def get_training_data_set(num, one_row=False):
    # param is file path and sample counts
    image_loader = ImageLoader('train-images.idx3-ubyte', num)
    label_loader = LabelLoader('train-labels.idx1-ubyte', num)
    return image_loader.load(one_row), label_loader.load()


# Acquire tested collection
# one_row represent if Trans to ROW VECTOR
def get_test_data_set(num, one_row=False):
    # param is file path and sample counts
    image_loader = ImageLoader('t10k-images.idx3-ubyte', num)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', num)
    return image_loader.load(one_row), label_loader.load()


# Trans 784 row vector to print
def print_img(inn_pic):
    inn_pic = inn_pic.reshape(28, 28)
    for i in range(28):
        for j in range(28):
            if inn_pic[i, j] == 0:
                print('  ', end='')
            else:
                print('* ', end='')
        print('')


if __name__ == "__main__":
    # Load the train data collection
    # After one-hot code sample label data collection
    train_data_set, train_labels = get_training_data_set(100)
    # Simplify the pic to black
    # .astype(bool).astype(int)
    train_data_set = np.array(train_data_set)
    train_labels = np.array(train_labels)
    # Fetch a sample
    one_pic = train_data_set[12]
    # Print the picture
    print_img(one_pic)
    # Print the label
    print(train_labels[12].argmax())
