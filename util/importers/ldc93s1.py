import wave
import random
import shutil
import threading
import numpy as np

from os import path
from os import rmdir
from os import remove
from glob import glob
from math import ceil
from Queue import Queue
from os import makedirs
from itertools import cycle
from os.path import getsize
from threading import Thread
from Queue import PriorityQueue
from util.gpu import get_available_gpus
from util.text import texts_to_sparse_tensor
from tensorflow.python.platform import gfile
from util.audio import audiofile_to_input_vector
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test

class DataSet(object):
    def __init__(self, graph, txt_files, thread_count, batch_size, numcep, numcontext):
        self._graph = graph
        self._numcep = numcep
        self._batch_queue = Queue(2 * self._get_device_count())
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()
        self._start_queue_threads()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def _start_queue_threads(self):
        batch_threads = [Thread(target=self._populate_batch_queue) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()

    def _create_files_circular_list(self):
        priorityQueue = PriorityQueue()
        for txt_file in self._txt_files:
          wav_file = path.splitext(txt_file)[0] + ".wav"
          wav_file_size = getsize(wav_file)
          priorityQueue.put((wav_file_size, (txt_file, wav_file)))
        files_list = []
        while not priorityQueue.empty():
            priority, (txt_file, wav_file) = priorityQueue.get()
            files_list.append((txt_file, wav_file))
        return cycle(files_list)

    def _populate_batch_queue(self):
        with self._graph.as_default():
            while True:
                n_steps = 0
                sources = []
                targets = []
                for index, (txt_file, wav_file) in enumerate(self._files_circular_list):
                    if index >= self._batch_size:
                        break
                    next_source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
                    if n_steps < next_source.shape[0]:
                        n_steps = next_source.shape[0]
                    sources.append(next_source)
                    with open(txt_file) as open_txt_file:
                        original = ' '.join(open_txt_file.read().strip().lower().split(' ')[2:]).replace('.', '')
                        targets.append(original)
                target = texts_to_sparse_tensor(targets)
                for index, next_source in enumerate(sources):
                    npad = ((0,(n_steps - next_source.shape[0])), (0,0))
                    sources[index] = np.pad(next_source, pad_width=npad, mode='constant')
                source = np.array(sources)
                self._batch_queue.put((source, target))

    def next_batch(self):
        source, target = self._batch_queue.get()
        return (source, target, source.shape[1])

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) /float(self._batch_size)))


def read_data_sets(graph, data_dir, batch_size, numcep, numcontext, thread_count=1):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    _ = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")

    # Conditionally extract LDC93S1 data
    LDC93S1_DIR = "LDC93S1"
    _maybe_extract(data_dir, LDC93S1_DIR, local_file, ["dev", "test", "train"])

    # Create dev DataSet
    train = dev = test = _read_data_set(graph, data_dir, LDC93S1_DIR, "dev", thread_count, batch_size, numcep, numcontext)

    # Create test DataSet
    ##test = _read_data_set(graph, data_dir, LDC93S1_DIR, "test", thread_count, batch_size, numcep, numcontext)

    # Create train DataSet
    ##train = _read_data_set(graph, data_dir, LDC93S1_DIR, "train", thread_count, batch_size, numcep, numcontext)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_extract(data_dir, extracted_data, archive, sets):
    wav_file = archive
    txt_file = archive.replace(".wav", ".txt")
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(path.join(data_dir, extracted_data)):
        for set in sets:
            makedirs(path.join(data_dir, extracted_data, set))
            shutil.copy(wav_file, path.join(data_dir, extracted_data, set))
            shutil.copy(txt_file, path.join(data_dir, extracted_data, set))

def _read_data_set(graph, data_dir, extracted_data, data_set, thread_count, batch_size, numcep, numcontext):
    # Create wav dir
    wav_dir = path.join(data_dir, extracted_data, data_set)

    # Obtain list of txt files
    txt_files = glob(path.join(wav_dir, "*.txt"))

    # Let us generate one batch for each thread that is going to run
    txt_files = [ txt_files[0] for x in xrange(batch_size * 2) ]

    print("I CAN HAZ txt_files=", len(txt_files), txt_files)

    # Return DataSet
    return DataSet(graph, txt_files, thread_count, batch_size, numcep, numcontext)
