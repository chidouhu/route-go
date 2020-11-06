import time
import tensorflow as tf
from classify_dnn import Classify_DNN
from train_new import train
from sample import sample
from cal_accuracy_v2 import acc
from road import RoadNetwork

r = 0
R = 30
context_file = "./indexed_nextlink.txt.20180627"
road = RoadNetwork(filename=context_file)
road.load()
origin_path = "./train_traj_data_clean"
target_path = "./rl_classify_train_data"
test_path = "./test_classify_data/10"
model_load_path = "./saver_base_1"
save_path = "./saver_rl_1/model"

dnn = Classify_DNN(n_feature=5, learning_rate=0.0001, vocab_size=2000000, embedding_size=8)

saver = tf.train.Saver()
model_path = tf.train.latest_checkpoint(model_load_path)
saver.restore(dnn.sess, model_path)
print(model_path)

while r<R:
    r += 1
    print("#########################")
    print(r)
    print("#########################")
    t1 = time.time()
    sample(dnn, road, origin_path, target_path, r)
    t2 = time.time()
    dnn = train(dnn, target_path, save_path, r)
    t3 = time.time()
    print("test acc")
    if r%3==0:
        acc(dnn, road, test_path)
    t4 = time.time()
    print(r, "epoch cost time", t4-t1, "sample cost", t2-t1, "train cost", t3-t2, "acc cost", t4-t3)

