import os
from road import *
import tensorflow as tf

def sample(dnn, road, origin_data, target_data, num_epoch):
    print("sampling ")
    print("-----------------------------------------------")
    file_names = os.listdir(origin_data)
    file_queue = [os.path.join(origin_data,x) for x in file_names]
    
    for file_ in file_queue:
        f = open(file_, "r")
        suffix = file_.split("/")[-1]
        new_file = target_data + "/" + (suffix + "_" + str(num_epoch))
        fw = open(new_file, "a")
        cnt = 0
        buffer = ""
        while True:
            line = f.readline()
            if not line:
                break
            order_id, start_pos, end_pos, traj = line.strip().split()
            traj = traj.split(",")
            traj = [int(x) for x in traj]
            start_link = traj[0]
            end_link = traj[-1]
            pre_link = 1999999
            for i,link in enumerate(traj[:-1]):
                next_links = road.get_all_next_links(link)
                if len(next_links)>1:
                    feature_batch = []
                    for next_link in next_links:
                        feature = [start_link, end_link, link, pre_link, next_link]
                        feature_batch.append(feature)
                    scores = dnn.get_prob(feature_batch)[:,1]
                    scores = [x for x in scores]
                    index = scores.index(max(scores))
                    predict_link = next_links[index]
                    if predict_link!=traj[i+1]:
                        data_write = str(start_link) + " " +str(end_link) + " " + str(link) + " " + str(pre_link) + " " + str(traj[i+1]) + " " + "1" + "\t"
                        pad = str(start_link) + " " + str(end_link) + " " + str(link) + " " + str(pre_link) + " " + str(predict_link) + " " + "0" + "\n"
                        data_write += pad
                        buffer += data_write
                        cnt += 1
                        if cnt%10000==0:
                            fw.write(buffer)
                            buffer = ""
                        break
                pre_link = link

        fw.write(buffer)
        f.close()
        fw.close()

