# -*- coding: UTF-8 _*_

class RoadNetwork(object):
    def __init__(self, filename):
        self.filename = filename
        self.roadnetwork = {}

    def load(self):
        file = open(self.filename)
        while True:
            line = file.readline()
            if not line:
                break
            fields = line.rstrip("\n").split("\t")
            linkid = int(fields[0])
            nextids = []
            if fields[1] == '':
                continue
            nextlinkids = fields[1].split(",")
            for id in nextlinkids:
                nextids.append(int(id))
            self.roadnetwork[linkid] = nextids

        file.close()

    def get_next_link_size(self, linkid):
        ids = self.roadnetwork.get(linkid)
        if ids == None:
            return 0
        else:
            return len(ids)

    def get_next_link(self, linkid, direction):
        ids = self.roadnetwork.get(linkid)
        if ids == None:
            return None
        else:
            if direction >= len(ids):
                return None
            else:
                return ids[direction]

    def get_all_next_links(self, linkid):
        return self.roadnetwork.get(linkid)
