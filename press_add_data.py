# -- coding: utf-8 --

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import gzip
import os
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import random
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, rnds, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            print("images.shape: %s labels.shape: %s rnds.shape: %s" % (images.shape, labels.shape, rnds.shape))
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert images.shape[3] == 1
            # images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            # images = numpy.multiply(images, 1.0 / 255.0)
            images = (images - images.min()) / (images.max() - images.min())
        self._images = images
        self._labels = labels
        self._rnds = rnds
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def rnds(self):
        return self._rnds

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange()]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._rnds = self._rnds[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._rnds[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()
    VALIDATION_SIZE = 500
    '''
	if fake_data:
		data_sets.train = DataSet([], [], fake_data=True)
		data_sets.validation = DataSet([], [], fake_data=True)
		data_sets.test = DataSet([], [], fake_data=True)
		return data_sets
	TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
	TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
	TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
	TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

	local_file = maybe_download(TRAIN_IMAGES, train_dir)
	train_images = extract_images(local_file)
	local_file = maybe_download(TRAIN_LABELS, train_dir)
	train_labels = extract_labels(local_file, one_hot=one_hot)
	local_file = maybe_download(TEST_IMAGES, train_dir)
	test_images = extract_images(local_file)
	local_file = maybe_download(TEST_LABELS, train_dir)
	test_labels = extract_labels(local_file, one_hot=one_hot)
	'''

    wav_list = []
    label_list = []
    # fo = open("rnd.txt", "rb")
    # str = fo.read()
    rnd_list = []
    # for i in range(len(str.split(","))):
    #     rnd_list.append(float(str.split(",")[i]))
    reader = csv.reader(open("data/wav2.csv", "rU"), delimiter=";")
    for wav, label, fr in reader:
        rnd = random.uniform(0, 1)
        rnd_arr = []
        rnd_arr.append(rnd)
        rnd_list.append(rnd_arr)

        wav_arr = []
        for i in range(len(wav.split(","))):
            wav_arr.append(float(wav.split(",")[i]))
        for i in range(2048):
            wav_arr.append(rnd)
        wav_arr.append(float(fr)*rnd)
        # wav_arr.append(float(fr))
        wav_list.append(wav_arr)

    wav_fr_list = np.array(wav_list)
    np.random.shuffle(wav_fr_list)
    wav_list_random = wav_fr_list[:, 0:512]   # 取出wav shape(,512)
    rnd_list = wav_fr_list[:, 512:2560]  # 取出rnd shape(,1)
    label_list_random = wav_fr_list[:, 2560:]  # 取出fr shape(,1)

    # wav归一化
    wav_list_normal = []
    for i in range(len(wav_list_random)):

        wav_max = np.max(wav_list_random[i], axis=0)
        wav_min = np.min(wav_list_random[i], axis=0)
        wav_list_random[i] = (wav_list_random[i] - wav_min) / (wav_max - wav_min)

        wav_list_normal.append(wav_list_random[i])
    # print (wav_list_normal)

    # label one-hot化
    label_list_rnd = []
    for i in range(len(label_list_random)):
        label_arr = np.zeros(100)
        label_arr[int(round(label_list_random[i]))]=1
        label_list_rnd.append(label_arr)
    # print (label_list_rnd)

    print('There are %d wav\nThere are %d label' % (len(wav_list), len(label_list)))

    train_images = numpy.array(wav_list_normal)
    # train_images = numpy.array(wav_list_random)
    train_images.reshape(-1, 512)
    train_labels = numpy.array(label_list_rnd)
    # train_labels = numpy.array(label_list_random)
    train_labels.reshape(-1, 1)
    train_rnd = numpy.array(rnd_list)
    # train_rnd = train_rnd[:, numpy.newaxis]
    print(train_images)
    print(train_images.shape)
    print (train_labels)
    print (train_labels.shape)
    print (train_rnd)
    print (train_rnd.shape)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    validation_rnd = train_rnd[:VALIDATION_SIZE]

    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    train_rnd = train_rnd[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_images, train_rnd, train_labels)
    data_sets.validation = DataSet(validation_images, validation_rnd, validation_labels)

    test_images = validation_images
    test_labels = validation_labels
    test_rnd = validation_rnd
    data_sets.test = DataSet(test_images, test_rnd, test_labels)
    return data_sets


# data_set = read_data_sets('tem')