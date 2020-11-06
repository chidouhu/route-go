from __future__ import print_function
import sys
import os
import tensorflow as tf
from classify import Classify_Model
from road import RoadNetwork
from train_model import train
from cal_accuracy import count_acc
from beam_search import beam_search
import time
import config as cf

data_path = "./train_traj_data_indexed/"
test_path = "./test_traj_data/201807w3"
#test_path = 'test_temp'

CONTEXT_FILE = "./indexed_nextlink.txt.20180719"
road = RoadNetwork(filename=CONTEXT_FILE)
road.load()
model_lstm = Classify_Model(learning_rate = cf.learning_rate, vocab_size = cf.vocab_size, embedding_size = cf.embedding_size)

save_dir = './model_saver/'
isExists = os.path.exists(save_dir)
if not isExists:
    os.makedirs(save_dir)
save_path = save_dir + 'model'

method = sys.argv[1]
print ("method", method)

if method == "train":
    for r in range(10):
        print('#############################')
        print("Epoch", r, time.asctime(time.localtime(time.time())))
        print('#############################')
        t1 = time.time()
        model_lstm = train(model_lstm, data_path, save_path, r, road)
        t2 = time.time()
        if r >= 0:
            print("testset acc: ", end = '')
            count_acc(model_lstm, road, test_path)
        t3 = time.time()
        print("train cost time", t2-t1, "acc cost time", t3-t2)
elif method == "test":
    epoch = sys.argv[2]
    saver = tf.train.Saver()
    saver.restore(model_lstm.sess, './model_saver/model-' + epoch)
    print("Epoch", epoch, "testset acc: ", end = '')
    count_acc(model_lstm, road, test_path)
elif method == "beam":
    epoch = sys.argv[2]
    K = int(sys.argv[3])
    saver = tf.train.Saver()
    model_file = './model_saver/model-' + epoch
    saver.restore(model_lstm.sess, model_file)
    print ("Load", model_file, "done!")
    beam_search(model_lstm, road, test_path, K)
else:
    print ("Unknown method !!!")

