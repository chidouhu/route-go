import math
import time
import sys
from road import *
import config as cf
import tensorflow as tf

#max decision num
maxlen = 150

link_length = {}
for l in open("./indexed_link_attr"):
    ll = l.strip().split("\t")
    link_length[int(ll[0])] = float(ll[1])

def get_link_length(linkid):
    if linkid not in link_length:
        return 8.6
    else:
        return link_length[linkid]

def isTraj(traj, route):
    if len(traj)!=len(route):
        return False
    for i in range(len(traj)):
        if int(traj[i])!=route[i]:
            return False
    return True

def isGetEnd(traj, route):
    if int(traj[-1])==route[-1]:
        return True
    else:
        return False

def beam_search(model, road, file_name, K):
    get_end_num = 0
    is_traj_num = 0
    hit_sim100_num = 0
    num = 0
    t1 = time.time()

    init_state = [0.0] * cf.lstm_hidden
    for l in open(file_name):
        ll = l.strip().split()
        order_id = ll[0]
        traj = [int(link) for link in ll[1].split(",")]
        start_link = traj[0]
        end_link = traj[-1]

        queue = [[[start_link], 0, init_state, init_state, get_link_length(start_link)]]
        cnt = 0
        min_length = 9999999
        done_list = []
        while cnt < maxlen and len(done_list) < K and len(queue) > 0:
            x_traj = []
            x_traj_len = []
            x_cross = []
            x_start = []
            x_end = []
            x_state_c = []
            x_state_h = []

            all_candidates = []
            split_batch = []
            start_pos = 0
            unfinished = 0
            for i in range(len(queue)):
                seq, prob, state_c, state_h, length = queue[i]
                cur_link = seq[-1]
                if length > 1.5 * min_length:
                    continue
                next_links = road.get_all_next_links(cur_link)
                if next_links == None:
                    print ("ERRORRRRRRRRRRRRRRRRR Miss", cur_link)
                    continue

                new_seq = []
                new_cross = []
                if len(next_links)==1:
                    while cur_link != end_link and next_links and len(next_links) == 1:
                        # avoid circle
                        if next_links[0] in seq + new_seq:
                            next_links = None
                            break
                        new_seq.append(next_links[0])
                        new_cross.append(cf.MISS)
                        length = length + get_link_length(next_links[0])
                        cur_link = next_links[0]
                        next_links = road.get_all_next_links(cur_link)
                    if cur_link == end_link:
                        candidate = [seq + new_seq, prob, state_c, state_h, length]
                        #all_candidates.append(candidate)
                        min_length = min(min_length, length)
                        done_list.append(candidate)
                        continue
                    if next_links == None:
                        continue

                if end_link in next_links:
                    candidate = [seq + new_seq + [end_link], prob, state_c, state_h, length + get_link_length(end_link)]
                    min_length = min(min_length, length)
                    done_list.append(candidate)
                    continue

                c = 0
                for next_link in next_links:
                    if next_link in seq:
                        continue
                    x_traj.append([cur_link] + new_seq)
                    x_traj_len.append(1 + len(new_seq))
                    x_cross.append([next_link] + new_cross)
                    x_start.append(start_link)
                    x_end.append(end_link)
                    x_state_c.append(state_c)
                    x_state_h.append(state_h)

                    candidate = [seq + new_seq + [next_link], prob, state_c, state_h, length + get_link_length(next_link)]
                    all_candidates.append(candidate)
                    c += 1
                split_batch.append((start_pos, start_pos + c))
                start_pos += c
                if c != 0:
                    unfinished += 1

            if unfinished == 0:
                break

            max_len = max(x_traj_len)
            traj_vec = []
            cross_vec = []
            for j in range(len(x_traj)):
                traj_vec.append(x_traj[j] + [0] * (max_len - x_traj_len[j]))
                cross_vec.append(x_cross[j] + [cf.MISS] * (max_len - x_traj_len[j]))
            (prob, out_state_c, out_state_h) = model.get_prob(traj_vec, x_traj_len, cross_vec, x_start, x_end, x_state_c, x_state_h)
            probs = []
            for cur_prob in prob:
                probs.append(cur_prob[0][1])

            for i in range(unfinished):
                (s, e) = split_batch[i]
                sum_prob = sum(probs[s : e])
                for j in range(s, e):
                    new_prob = probs[j] / sum_prob
                    all_candidates[j][1] += math.log(new_prob)
                    all_candidates[j][2] = out_state_c[j]
                    all_candidates[j][3] = out_state_h[j]
            if len(all_candidates) > K - len(done_list):
                all_candidates.sort(key=lambda tup:tup[1], reverse=True)
            queue = all_candidates[:K - len(done_list)]
            cnt += 1
        #print (order_id, cnt)
        arrive_num = 0
        sim100_num = 0

        num += 1
        for res in done_list:
            route = res[0]
            if isGetEnd(traj, route):
                arrive_num += 1
            if isTraj(traj, route):
                sim100_num += 1
        if len(done_list) > 0:
            prob = -100000.0
            for i in done_list:
                if i[1] > prob:
                    prob = i[1]
                    route = i[0]

            if isGetEnd(traj, route):
                get_end_num += 1
            if isTraj(traj, route):
                is_traj_num += 1
            if sim100_num > 0:
                hit_sim100_num += 1
        if num % 1000 == 0:
            print ("Size", num, "Top 1 sim100", '%.2f%%' % (is_traj_num / num * 100), "Top", K, "sim100", '%.2f%%' % (hit_sim100_num / num * 100), "get_end_ratio", '%.2f%%' % (get_end_num / num * 100),  "cost", '%.2f' % (time.time() - t1))

    print("Finally")
    print("Top 1 sim100", '%.2f%%' % (is_traj_num / num * 100))
    print("Top", K, "sim100", '%.2f%%' % (hit_sim100_num / num * 100))
    print("get_end_ratio", '%.2f%%' % (get_end_num / num * 100))
    print("cost time ", '%.2f' % (time.time() - t1))

