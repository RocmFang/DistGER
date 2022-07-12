import numpy as np
import random
from collections import Counter

class Graph():

    def __init__(self, nx_G, is_directed):
        self.G = nx_G
        self.is_directed = is_directed

    def huge_walk(self, start_node, comm_neighbor, R, min_L):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        walk = [start_node]
        ent_walk_list = []
        walk_length_list = []
        r2 = 1
        r = 1

        obj = {start_node: 1}

        while r2 > R and r > 0:

            cur = walk[-1]

            cur_nbrs = list(G.neighbors(cur))
            # print cur, cur_nbrs

            if len(cur_nbrs) > 0:

                src_deg = G.degree(cur)
                dst = random.choice(cur_nbrs)
                dst_deg = G.degree(dst)
                # print 'dst node:', dst

                walk, ent_walk, obj = self.walk_path(walk=walk, cur=cur, dst=dst, src_deg=src_deg, dst_deg=dst_deg,
                                                     comm_neighbor=comm_neighbor, obj=obj)

                ent_walk_list.append(np.log10(ent_walk))

                walk_length = len(ent_walk_list)

                walk_length_list.append(np.log10(walk_length))

                if walk_length > min_L:

                    if len(set(ent_walk_list)) > 1:

                        corrcoef = np.corrcoef(walk_length_list, ent_walk_list)
                        r = corrcoef[0][1]
                        r2 = r**2

                    else:
                        r = 0
                        r2 = 0
            else:

                break

        return walk


    def simulate_walks(self, nodes, comm_neighbor, r, min_L, h):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        walk_length_list = []
        H = [0]
        delt_h = 1
        i = 0

        deg_data = dict(G.degree())
        sum_deg = sum(deg_data.values())
        p_deg = {}

        for key in deg_data.keys():

            if deg_data[key] > 0:
                p_deg[key] = (deg_data[key])/float(sum_deg)

        while delt_h > h:

            random.shuffle(nodes)
            p_deg_list = p_deg

            for node in nodes:

                walk = self.huge_walk(start_node=node, comm_neighbor=comm_neighbor, R=r, min_L=min_L)

                walks.append(walk)
                walk_length = len(walk)
                walk_length_list.append(walk_length)

            H.append(self.relative_entropy_text(walks, p_deg_list))
            delt_h = abs(H[-1]-H[-2])

            i = i+1

            print i, ':', round(delt_h, 6)

        return walks, walk_length_list


    def walk_path(self, walk, cur, dst, src_deg, dst_deg, comm_neighbor, obj):

        ncn = comm_neighbor[cur, dst]

        p = random.random()

        p_acp = (1 / float(src_deg - ncn)) * (float(max(src_deg, dst_deg)) / min(src_deg, dst_deg))

        p_norm = np.tanh(p_acp)

        if p_norm > p:
            walk.append(dst)
            ent, obj = self.entropy_walk_uodate(obj, dst, None)
        else:
            walk.append(dst)
            walk.append(cur)
            ent, obj = self.entropy_walk_uodate(obj, dst, cur)

        return walk, ent, obj

    def entropy_walk_uodate(self, obj, vertex1, vertex2):

        if vertex2 is None:

            if obj.get(vertex1) is not None:
                obj[vertex1] = obj[vertex1] + 1
            else:
                obj[vertex1] = 1

        else:

            if obj.get(vertex1) is not None:
                obj[vertex1] = obj[vertex1] + 1
                obj[vertex2] = obj[vertex2] + 1
            else:
                obj[vertex1] = 1
                obj[vertex2] = obj[vertex2] + 1

        sum_num = sum(obj.values())
        ent = 0.000000000001

        for word in obj.values():
            p_v = float(int(word)) / sum_num
            ent -= p_v * np.log2(p_v)

        return ent, obj

    def relative_entropy_text(self, walks, p_deg):

        word_text = []

        for path in walks:

            for word in path:
                word_text.append(word)

        obj = Counter(word_text)

        sum_text = sum(obj.values())

        H = 0.000000001

        for key in p_deg.keys():

            p_word = int(obj[key]) / float(sum_text)
            H += p_deg[key] * np.log(p_deg[key] / p_word)

        return H
