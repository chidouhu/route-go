from road import *
import numpy as np
import config as cf
import copy

def count_acc(model, road, file_name):
    cnt = 0
    right = 0

    data = []
    for line in open(file_name):
        ll = line.strip().split(" ")
        oid = ll[0]
        traj = [int(link) for link in ll[1].split(",")]
        data.append(traj)
    data.sort(key=lambda x: len(x))

    traj_num = len(data)
    batch_size = 8192
    batch_num = int(traj_num / batch_size) + 1

    for i in range(batch_num):
        #print (i , batch_num)
        s = batch_size * i
        e = min(batch_size * (i + 1), traj_num)

        trajs = data[s : e]
        traj_lens = [len(j) for j in trajs]
        max_len = max(traj_lens)

        state_c_vec = [[0.0] * cf.lstm_hidden for m in range(e - s)]
        state_h_vec = [[0.0] * cf.lstm_hidden for m in range(e - s)]

        for j in range(max_len - 1):
            x_traj = []
            x_traj_len = []
            x_cross = []
            x_start = []
            x_end = []
            x_state_c = []
            x_state_h = []
    
            size_batch = []
            label_batch = []
            cur_batch = []
            for cur in range(e - s):
                cur_traj = trajs[cur]
                if j > len(cur_traj) - 2:
                    continue
                cur_link = cur_traj[j]
                next_links = road.get_all_next_links(cur_link)
                if next_links == None:
                    print ("ERRORRRRRRRRRRRRRRRRR Miss", cur_link)
                if cur_traj[j + 1] not in next_links:
                    print ("ERRORRRRRRRRRRRRRRRRR Wrong", cur_link, cur_traj[j + 1])
                for cc in next_links:
                    x_traj.append([cur_link])
                    x_traj_len.append(1)
                    x_cross.append([cc])
                    x_start.append(cur_traj[0])
                    x_end.append(cur_traj[-1])
                    x_state_c.append(state_c_vec[cur])
                    x_state_h.append(state_h_vec[cur])
                size_batch.append(len(next_links))
                label = next_links.index(cur_traj[j + 1])
                label_batch.append(label)
                cur_batch.append(cur)

            (prob, out_state_c, out_state_h) = model.get_prob(x_traj, x_traj_len, x_cross, x_start, x_end, x_state_c, x_state_h)
            scores = []
            for cur_prob in prob:
                scores.append(cur_prob[0][1])

            score_s = 0
            for cur in range(len(label_batch)):
                state_c_vec[cur_batch[cur]] = out_state_c[score_s]
                state_h_vec[cur_batch[cur]] = out_state_h[score_s]
                score_e = score_s + size_batch[cur]
                if size_batch[cur] != 1:
                    cur_score = scores[score_s: score_e]
                    y = np.argmax(cur_score)
                    label = label_batch[cur]
                    if y == label:
                        right += 1
                    cnt += 1
                score_s = score_e

    accuracy = right / cnt
    print ("count,", cnt, "accuracy", accuracy)
    return accuracy

