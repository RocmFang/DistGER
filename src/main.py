import argparse
import networkx as nx
import HuGE as huge
from gensim.models import Word2Vec
import time


def parse_args():
    '''
    Parses the HuGE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run HuGE.")

    parser.add_argument('--input', nargs='?', default='../graph/CA-AstroPh.txt',
                        help='Input graph path')

    parser.add_argument('--comnb', nargs='?', default='../pre_data/CA-AstroPh_comneig.txt',
                        help='Preprocessing for common neighbors')

    parser.add_argument('--output', nargs='?', default='../emb/CA-AstroPh.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--sample', type=float, default=0.5,
                        help='Inout hyperparameter. Default is 0.5')

    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers. Default is 10.')

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.set_defaults(undirected=False)

    parser.add_argument('--r', type=float, default=0.999,
                        help='R square. Default is 0.999')

    parser.add_argument('--min_L', type=int, default=10,
                        help='The minimum walk length. Default is 10')

    parser.add_argument('--h', type=float, default=0.001,
                        help='Variation H. Default is 0.001')


    return parser.parse_args()

def read_graph():
    '''
    Reads the input network in networkx.
    '''

    G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.Graph())

    return G

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)

    return model.wv


def common_neighbor_loading():

    comm_neighbor = {}

    with open(args.comnb, 'r') as f:
        com_edges = f.readlines()

        for edge in com_edges:

            com_ind = edge.split()
            comm_neighbor[int(com_ind[0]), int(com_ind[1])] = int(com_ind[2])

    return comm_neighbor


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    print "loading graph"
    load_graph_time = time.time()

    nx_G = read_graph()

    G = huge.Graph(nx_G, args.directed)

    loaded_graph_time = time.time()

    print "load graph completed! load time used: ", (loaded_graph_time - load_graph_time), 's'

    print 'Common Neighbor Loading:'

    time1 = time.time()

    comm_neighbor = common_neighbor_loading()

    time2 = time.time()

    print "common neig loading time:", (time2 - time1), 's'

    nodes = list(G.G.nodes())

    print "Random walk starting:"

    walks_time = time.time()

    walks, walk_length_list = G.simulate_walks(nodes, comm_neighbor, args.r, args.min_L, args.h)

    walks_time_end = time.time()

    print "Walk path completed! time used:", (walks_time_end - walks_time), 's'

    learn_time = time.time()

    wv = learn_embeddings(walks)

    learn_time_end = time.time()

    print 'learning time used:', (learn_time_end-learn_time), 's'


if __name__ == "__main__":
    args = parse_args()
    main(args)