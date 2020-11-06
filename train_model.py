import time
import random
import numpy as np
import datetime
import config as cf
from road import *

def train(model, train_traj_path, model_save_path, num_epoch, road):
    start_time = time.time()
    pred_day = datetime.datetime.strptime("20180709", '%Y%m%d')
    file_list = [train_traj_path + "%s" % (pred_day + datetime.timedelta(days=-i)).strftime('%Y%m%d') for i in
                      range(cf.train_days, 0, -1)]
    for filename in file_list:
        print (filename)
        data_traj = {}
        data_traj_len = {}
        data_cross = {}
        data_start = {}
        data_end = {}
        data_y = {}

        idx = 0
        idx_vec = []
        #drop = 0
        for line in open(filename):
            #drop += 1 # too many
            #if drop % 10 != 0:
            #    continue
            ll = line.strip().split(" ")
            oid = ll[0]
            traj = [int(link) for link in ll[1].split(",")]
            start = traj[0]
            end = traj[-1]
            traj_len = len(traj)
            cross1 = [cf.MISS] * traj_len
            cross2 = [cf.MISS] * traj_len
            y1 = [0] * traj_len
            y2 = [0] * traj_len

            pos_or_neg = 0
            for i in range(0, traj_len - 1):
                next_links = road.get_all_next_links(traj[i])
                if next_links == None or len(next_links) == 1 or traj[i + 1] not in next_links:
                    continue
                pos_link = traj[i + 1]
                neg_link = random.choice(list(filter(lambda x: x != pos_link, next_links)))
                # avoid y = [1,1,1,1,1,...]
                if pos_or_neg == 0:
                    cross1[i] = pos_link
                    y1[i] = 1
                    cross2[i] = neg_link
                    y2[i] = 0
                else:
                    cross1[i] = neg_link
                    y1[i] = 0
                    cross2[i] = pos_link
                    y2[i] = 1
                pos_or_neg = 1 - pos_or_neg

            data_traj[idx] = traj
            data_traj_len[idx] = traj_len
            data_cross[idx] = cross1
            data_start[idx] = start
            data_end[idx] = end
            data_y[idx] = y1
            idx_vec.append(idx)
            idx += 1

            data_traj[idx] = traj
            data_traj_len[idx] = traj_len
            data_cross[idx] = cross2
            data_start[idx] = start
            data_end[idx] = end
            data_y[idx] = y2
            idx_vec.append(idx)
            idx += 1
        print ("load done")

        np.random.shuffle(idx_vec)

        batch_size = cf.train_batch_size
        num = int(len(idx_vec) / batch_size) + 1
        print ("batch num:", num)

        for i in range(num):
            s = i * batch_size
            e = int(min((i + 1) * batch_size, len(idx_vec)))

            init_state = [[0.0] * cf.lstm_hidden for m in  range(e - s)]

            x_traj = [data_traj[idx] for idx in idx_vec[s : e]]
            x_traj_len = [data_traj_len[idx] for idx in idx_vec[s : e]]
            x_cross = [data_cross[idx] for idx in idx_vec[s : e]]
            x_start = [data_start[idx] for idx in idx_vec[s : e]]
            x_end = [data_end[idx] for idx in idx_vec[s : e]]
            y = [data_y[idx] for idx in idx_vec[s : e]]
            max_len = max(x_traj_len)
            traj_vec = []
            cross_vec = []
            y_vec = []
            for j in range(e - s):
                traj_vec.append(x_traj[j] + [0] * (max_len - x_traj_len[j]))
                cross_vec.append(x_cross[j] + [cf.MISS] * (max_len - x_traj_len[j]))
                y_vec.append(y[j] + [0] * (max_len - x_traj_len[j]))

            loss = model.train(traj_vec, x_traj_len, cross_vec, x_start, x_end, init_state, init_state, y_vec)
            #print (i, "loss", loss)
        print ("train done", time.time() - start_time)

    end_time = time.time()
    print("TOTAL COST TIME: ", end_time - start_time)
    model.save_model(model_save_path,num_epoch)
    print("save the model successfully")
    return model

