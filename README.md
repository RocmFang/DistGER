<meta name="robots" content="noindex">

# The longer response for the reivews of DistGER paper

## Review 1
#### <p align="justify">  O1. Related work not adequately represented and compared to. The paper presents itself as a scalable approach for obtaining graph embeddings, but it neither discusses nor compares to appropriate alternatives. One example is Gosh [A], which reports faster runtimes and better AUCROC scores on less hardware, but is not mentioned in the paper. Other scalable approaches are cited but not compared to, e.g., VERSE [51]. Ideas such as reusing negative samples or communicating "hot" items differently are reminiscent of multi-technique parameter servers such as NuPS [B]. There is also a Huge+ paper [C], which hasn't been discussed or cited.</p>
 
#### Thanks for your suggestions. We will supplement the discussion of related approaches and compare them as detailed below in the revision.
- <p align="justify"> Since one of the motivations of our work is that there is yet no end-to-end distributed system to support graph embedding via information-oriented random walks, the competitors we selected are state-of-the-art distributed graph embedding systems.</p>
- <p align="justify">  The reason for not comparing with VERSE is that it proposes a sequential scalable method and does not have a distributed implementation. We evaluated VERSE on Youtube graph (|E|=3M), it takes more than 12 hours, while DistGER does it in a minute. DistGER also provides a general API for random walk tasks that can support VERSE in a distributed setting. If given a chance to revise, we shall demonstrate more generality of DistGER by implementing VERSE on it.</p>
- <p align="justify">  Although GOSH implemented a GPU-based single machine system which trains on a number of smaller and coarsened graphs, [r1] reports that GOSH crashes due to exhausting the available hardware resources on Twitter graph, and the downstream task accuracy of GOSH is far behind even our competitor PBG on the same evaluated graphs, while DistGER achieves more than 10% accuracy gains compared to PBG on average. For fairness, the competitors we selected and our DistGER – all use CPU architecture. In addition, the coarsening framework of GOSH and DistGER can mutually benefit each other. (Please see O6 of Reviewer 2).</p>
  [r1] Rychalska et.al., Cleora: A Simple, Strong and Scalable Graph Embedding Scheme. ICONIP 2021.
- <p align="justify">  The very recent approach (SIGMOD22) NuPS and the training component (DSGL) of DistGER both attempt to mitigate the effects of skewed frequency accesses and randomly sampled accesses, but the schemes they use are significantly different. For the skewed accesses, NuPS employs replication and relocation-based multi-technique parameters management to deal with hot and rare parameters, while DSGL replaces the full-synchronization with a synchronization block to update vectors across multiple machines. On Twitter, the full-synchronization model needs 85.2B messages over 8 machines, while DSGL just needs 46.3M messages. For the randomly sampled accesses, NuPS implements four sampling schemes to handle the negative sampling, but DSGL uses the multi-windows shared-negative samples mechanism which not only reuses negative samples, but also increases the batch matrix sizes to fully utilize CPU resources. Furthermore, DSGL also considers reusable positive samples. Finally, our global-matrix and the local-buffer schemes are proposed to improve the locality of parameter accesses and reduce the ping-ponging of cache lines across multiple cores. </p>
- <p align="justify">  HuGE+ is also a recent approach (IEEE TBD 2022), which is an extension of HuGE, it uses the same HuGE information-oriented method to determine the walk length and number of walks per node, thus the efficiency difference is not much in HuGE and HuGE+. DistGER’s general API  is compatible to run HuGE+. </p>
 
#### <p align="justify">  O2. Setup not convincing. The key premise of this paper is to partition the graph over a set of machines and then devise algorithms that minimize the communication costs that occur since random walks cross node boundaries. This setup is not convincing to me: even the largest graphs considered in the study easily fit on a single machine. A more straightforward and efficient approach is thus to simply replicate the graph and compute all random walks independently without communication. Whether parallelizing the node2vec algorithm across multiple machines is beneficial remains unclear, but an approach such as the one in [B] seems to be readily usable in such a setup.</p>
- <p align="justify"> Although the modern server can easily store most of the studied graphs in DRAM, such as the memory required for the Twitter graph is 26.2G, all our computations cannot be often done on a single machine due to the limited memory. For instance, in the sampling phase in KnightKing via Node2Vec, since each walker maintains a walker information (walker_id, step, node_id) at each step, the computation requires more than 798G memory, it would crash a single machine having lesser memory. Similarly, for the learning phase of DistGER, the space cost is O(2|V|d) (d is the embedding dimensionality, usually ranging from 128-1024), thus the required memory capacity is larger than the size of the loaded graph. Even if the memory capacity of a single machine can meet the computation requirements, simply replicating the whole graph on each machine will introduce low memory utilization, resulting in wasted memory and energy consumption problems.</p>
- <p align="justify">In DistGER, the information effectiveness measurement for multiple walks (defined in Eq.8 & 9) requires synchronizing the generated information. The training model needs to combine all generated walk information (such as frequency) to construct the global matrix and also requires synchronization of the parameters. Thus, communication cannot be avoided.</p>
 
#### <p align="justify"> O3. Experimental study not sufficiently insightful. The paper does not evaluate its many smaller contributions systematically, so that it's unclear where the performance benefits come from. Runtime and quality is evaluated separately so that it's not clear whether / to what extent parallelization affects quality and properties such as the number of epochs required for convergence. Related methods are run in their default configuration (e.g., r=10, L=80), whereas the proposed method appears tuned. Often, it's not really clear what is shown, e.g., how is the "end-to-end" time defined (e.g., it does not seem to include graph partitioning, and it's unclear when the method actually "ends").</p>
- <p align="justify"> We evaluated the individual part in Sec.6.5. (1) we evaluated the random walk efficiency of the proposed incremental information-computing mechanism. (2) To evaluate the benefit of our partition scheme, we employed our proposed scheme MPGP, the results show that it reduces (avg. reduction 45%) the number of cross-machine messages and improves the efficiency by 39.3% for the random walking procedure. We also exhibited the effect of different node streaming to our partition efficiency. (3) For the learning procedure (DSGL), we first evaluated the benefits of the information-centric random walk to training. Next, we conducted experiments under the same size of corpus to compare the training efficiency of DSGL over Pword2vec. We reported the CPU throughput in Sec.6.5 to demonstrate that the benefits compared to Pword2vec come from our multi-windows shared-negative samples computing mechanism. </p>
- <p align="justify">  For the sampling-based graph embedding model, the running time is determined by the walk length (L) and number of walks per node (r). As an information-centric approach, the best values of L and r are automatically decided based on information theoretic measurement, but we will also show how different values of r and L impact accuracy-running time tradeoff in the revision.</p>
- <p align="justify"> We will show more variations of parameters such as r and L,  they trade-off the running time vs accuracy, but we find that r=10 and L=80 produces good quality results in most cases for Deepwalk and node2vec, notice that they are fixed for all nodes in Deepwalk and node2vec for a graph. For the task effectiveness evaluations, all evaluated frameworks used a grid search over the training parameters to get the best results as described in the parameter setting (Sec. 6.1).</p>
- <p align="justify"> For all baselines, the end-to-end time in our experiments excludes its partition time. For the sampling-based systems, the end-to-end time refers to the running time of sampling and training, while for PytorchBigGraph and DistDGL it is only the training time.</p>
 
#### <p align="justify"> O4. Many heuristics, little analysis. The paper has a strong engineering focus. Most of its decisions appear reasonable, but also quite natural (e.g., incremental computation) or close to prior work (e.g., GSGL). The paper says little, if anything, on the theoretical properties of its methods and optimizations. Overall, the approach is quite involved, which I consider problematic given the rather narrow scope of this work.</p>
- <p align="justify">In HuGE and DistGER, the random walk parameters r and L are decided systematically based on a well-established notion of information theory. Incremental computation of information entropy is a novel contribution, and though natural, it helps in improving DistGER’s efficiency. We ensure its accuracy guarantee with theoretical proof (Eq.10-15). For partitioning, multi-windows shared-negative sampling computing, and hotness-block based synchronization, although we do not have any accuracy guarantee, their effectiveness has been verified through experiments, their time complexity analysis and efficiency improvement have been both theoretically and experimentally analyzed.</p>
- <p align="justify"> We make significant improvements in DSGL than prior works, (1) DSGL propose a novel computing mechanism to further increase the batch matrix sizes compared to Pword2vec. (2) DSGL does not use any additional structures to assist the weight computing compared to pSGNScc. In addition, DSGL also improves the access locality for the lifetime of a thread and reduces the ping-ponging of cache lines across multiple cores, not considered in prior works.</p>
- <p align="justify"> Most existing graph embedding methods suffer from computation-efficiency challenges with large-scale graphs. Although the scalable sequential approach HuGE resolves the routine random walk issue, it still requires more than one week for a billion-edge Twitter graph. Thus, we propose an end-to-end distributed system DistGER to support graph embedding via information-oriented random walks. Most importantly, it has good generality, it is not only designed for HuGE, but also provides an easy-to-operate API to support traditional methods, such as Deepwalk and node2vec, improving them from the efficiency and effectiveness problems due to the routine random walks (Fig.10). Meanwhile, DistGER exhibits better scalability than other systems (Fig. 6).</p>
 
#### <p align="justify"> O5. The proposed partitioning scheme appears to be so slow that it thwarts the performance benefits for larger graphs. E.g., the end-to-end-time reported for TW is less than 1000s, the partitioning time larger than 35000s.</p>
- <p align="justify"> The partition only runs once and the partition results can be reused for many embedding systems (and even for other graph analytic workloads, such as personalized PageRank). Moreover, to generate a better quality of embeddings, the models usually need to be tuned multiple times, thus the gains of partition amortize its overhead. Finally, the random walk efficiency can benefit from the partition scheme as shown in Sec. 6.5, our solution can deliver an average efficiency gain of 39.3%.</p>
- <p align="justify"> Our proposed partition scheme is more efficient than other methods including the METIS [28], LDG [47] and FENNEL [52], it is on average 12.89× faster than competitors (Table 4).</p>

#### <p align="justify"> 12. Availability: The code appears reasonable, but some things (e.g., partitioning) are not yet documented. </p>
- <p align="justify"> The partition code is located in src/tools and named as mpgp.cpp, please check. </p>

 
## Review2
#### <p align="justify"> O1. As it is not your contribution Algorithm 1 is not required to be present in the paper and can be removed.</p>
- <p align="justify">Algorithm 1 shows the walking procedure of node2vec in the walker-centric programming model, thus it serves as a preliminary to demonstrate how HuGE-D works on KnigtKing. If there is a space constraint, we will remove it in the revision.</p>
 
#### <p align="justify"> O2. What is the intuition behind PS1 and PS1. How γ is usually set?</p>
- <p align="justify">  PS1 denotes how many neighbors of an unpartitioned node are in the candidate partition, thus a higher value of PS1 implies that the unpartitioned node should be assigned to the candidate partition. PS2 is defined as the number of common neighbors between the unpartitioned node and the nodes in the candidate partition. Since the number of common neighbors is widely used to measure the similarity of node-pairs during random walk, considering a higher value of PS2 is more in line with the characteristics of random walk. </p>
- <p align="justify"> 𝛾 is a slack parameter that allows deviation from the exact load balancing. Although setting a strict 𝛾 will ensure the load-balancing, it may hamper the partition quality, resulting in lower utilization of the local partition. On the other hand, setting a larger 𝛾 will relax the load-balancing constraint, and would create skewed partitioning. Thus, there is a trade-off between the partition quality and load-balancing. In our case, we experimentally analyze those trade-offs and we set 𝛾 =2 since this provides a sweet point among those trade-offs. If given a chance, we shall add experimental results to show how different 𝛾 affects the quality and load balancing. </p>

#### <p align="justify"> O3. One more line is required to explain the main idea behind the Galloping algorithm for the paper to be self-complete.</p>
- <p align="justify"> We will add a brief explanation of the main idea for the Galloping algorithm in the revision.</p>
- <p align="justify"> There are two stages in the Galloping algorithm to iteratively pick an element 𝑆𝑎[𝑖] from the smaller set and search whether it exists in the larger set 𝑆𝑏. First stage explores a range by doubling the search index 𝑝 until for the first time an element 𝑆𝑏[𝑗 + 𝑝] less than 𝑆𝑎[𝑖] is encountered. The second stage simply leverages a binary search to find the exact position of 𝑆𝑏 in the range of [𝑗, 𝑗 + 𝑝]. This strategy avoids a large number of invalid comparisons and helps the pointer to the smaller element move quickly.</p>
 
#### <p align="justify"> O4. If I understand correctly the partitioning is performed sequentially for each node and not in parallel right (the computations for assigning the node are parallelized though)? This is not clear.</p>
- <p align="justify"> In the paper, partitioning is performed sequentially for each node, though the computations for assigning the node to the appropriate server are parallelized, our proposed partition scheme is more efficient than other methods including the METIS [28], LDG [47] and FENNEL [52], it is on average 12.89× faster than competitors (Table 4). If given a chance to revise, we will try to implement a parallel streaming partitioning algorithm, however it is challenging to ensure the quality and the efficiency parallel streaming partitioning as shown in: [Hua et al., Quasi-Streaming Graph Partitioning: A Game Theoretical Approach, IEEE TPDS, 2019].
  
#### <p align="justify"> O5. It seems that partitions generated can have significantly different sizes. Is this the case? Please comment.</p>
- <p align="justify"> Since our partition scheme MPGP not only reduces cross-machine communications, it also uses a dynamic workload constraint to ensure load-balancing, and thus the size of each partition is similar. For instance, to partition the Twitter graph on 8 machines, the number of nodes allocated to each partition is as follows: 5309435, 5181739, 5179742, 5122789, 5329900, 5151167, 5390088, 4987370. Meanwhile, MPGP achieves 43% efficiency improvement compared to only the workload-balancing scheme of KnightKing (Fig.8 (c, d)).</p>

#### <p align="justify"> O6. MILE [x1] is also a generic methodology (is using DeepWalk and Node2vec too) allowing contemporary graph embedding methods to scale to large graphs and coarsens the graph into smaller ones using a hybrid matching technique to maintain the backbone structure of the graph. It could be considered for related work as well.</p>
- <p align="justify">We will discuss this work in our new version. MILE uses a graph coarsening method to deal with larger graphs. This is orthogonal to the distributed approach as in DistGER and our other competitors. The source code given by MILE produces out-of-memory error for a larger UK graph (|E|=3.7B), but DistGER can perform the embedding of the UK graph in 1935s. Nevertheless, the coarsening framework of MILE and the distributed platform of DistGER can mutually benefit each other. In the revision, if possible, with their coarsening technique, we'll support ever bigger graphs in DistGER.</p>

#### <p align="justify"> O7. Scalability. It would be interesting to see scalability results for other datasets as well in an extended version of the paper online (or a technical report)</p>
- <p align="justify">  We will demonstrate more results in our new version or in a technical report. For instance, as the number of computing machines increases from 1 to 8, on Twitter graph, the end-to-end running times become 3090.1s, 1739.4s, 1196.9s, 745.8s, respectively, on Com-Orkut graph, the end-to-end running times are 303.8s, 203.5s, 149.3s, 88.7s, respectively. </p>

#### <p align="justify"> 13. Minor remarks: Please replace “in stead” with “instead” in section 3.2.</p>
- <p align="justify"> Thanks for pointing out our minor mistake, we will modify it in the new version. </p>


 
## Review3
#### <p align="justify"> W1. Almost the whole Introduction section is discussing related work. Almost no discussion of higher-level specific problem statements (i.e. why billion-node graphs are important) and motivation behind the solution (i.e. why the current solution is so important in context of a real-world problem, where a billion-node graph is a solution). I understand "As a preview, comparing with KnightKing, Pytorch-BigGraph, and DistDGL, our DistGER achieves 19.3×, 16.9×, and 118.9× faster embedding on average, and easily scales to billion-edge graphs", but it is still unclear why it is a "hair on fire" problem that the current solutions are slower and not good enough.</p>
 
- <p align="justify"> Our paper states the billion-edge graph (as done by all other related work) instead of the billion-node graph. We also plan to add experimental results on a billion-node graph (see reply to W2).</p>
- <p align="justify"> Graph embedding is widely used in downstream machine learning tasks. In these applications, graphs are usually huge, having millions of nodes and billions of edges. For instance, the Twitter graph includes over 41 million user nodes and over a billion edges; it has extensive requirements for link prediction and classification tasks. The graph of users and products at Alibaba also consists of more than two billion user-product edges, which forms a giant bipartite graph for its recommendation task. Natural language processing tasks take advantage of knowledge graphs, such as Freebase with 1.9 billion triples.</p>
- <p align="justify"> For the billion-edge graph Twitter, the current solutions either cannot be terminated in a reasonable time, or crash due to exhausting the available hardware resources. For instance, the DistDGL does not terminate in 1 day, while KnightKing triggers the issue of out-of-memory during sampling. Although Pytorch-BigGraph can be done in 2173s on 8 machines, it can be observed that on 4 machines it takes even less time – 1767s, indicating that the hardware resources cannot be fully utilized due to poor scalability which is consistent with Fig. 6. While our proposed DistGER only needs 745s with better task-accuracy and scalability.</p>

#### <p align="justify"> W2. The solution is positioned for billion-node graphs, but there is no such graphs among the datasets using in the experimental evaluation (Table 2)</p>
- <p align="justify"> Our paper states the billion-edge graph. The size of evaluated graphs we used is consistent with that used in all other related works. We have also tested on a larger, new billion-edge graph UK (3.7 billion edges) in a cluster with 8 machines, DistGER only needs 1935s.</p>
- <p align="justify"> In revision, we also plan to evaluate on a billion-node graph to further exhibit the scalability of our scheme.</p>

#### <p align="justify"> W3. Abstract: "The increasing availability of billion-edge graphs...". Some examples would strengthen the problem statement and motivation.</p>
- <p align="justify"> Thanks for your suggestion. We will add some examples in the new version as follows. The link prediction task on Twitter with over one billion edges and the product recommendation for the user-product network of Alibaba which has more than two billion user-product edges.</p>

#### <p align="justify"> W4. Figures 8-10: Some fonts are too small, which makes the legends illegible.</p>
- <p align="justify"> Thanks for your suggestion. We will modify the figure in the new version.</p>




***
***
## This codebase is for the paper: Distributed Graph Embedding with Information-Oriented Random Walks

# Prerequisites

- Ubuntu 16.04
- Linux kernel 4.15.0
- g++ 9.4.0
- CMake 3.10.2
- [MPICH 3.4.2](https://www.mpich.org)
- [MKL 2022.0.2](https://software.intel.com/en-us/mkl)

# Datasets

The evaluated dataset Youtube and LiveJournal are prepraed in the "dataset" directory.

Since the the space limited of the repository, the other datasets [Twitter](https://law.di.unimi.it/datasets.php), [Com-Orkut](https://snap.stanford.edu/) and [Flickr](http://datasets.syr.edu/pages/datasets.html) can be found in their open resource.

# Setup

First Compile DistGER with CMake:

```
mkdir build && cd build

cmake ..

make
```

Then the compiled application executable files are installed at the "bin" directory:

```
ls ./bin
```

# Partitioning

If we need to run the train data for the downstream tasks, such as Link prediction, the test data also should be processed.

```
cd build

./bin/mpgp -i [train_data] -e [test_data] -v [vertex_num] -p [partition_num] -t [float:0, integer:1]
```

The partitioned dataset will be saved in the input dataset directory. 

# Graph Embedding

To start the embedding, we fist need to cover the train graph to binary format

```
cd build

./bin/gconverter -i ../dataset/LJ.train-8-r -o ./LJ-8.data-r -s weighted
```

Then create the "out" directory to save the walks file or embedding file

```
mkdir out
```

### Run in Single-machine Environment
```
mpiexec -n 8 ./bin/huge_walk -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --make-undirected -o ./out/walks.txt -eoutput ./out/LJ-r_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5  -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 2
```

### Run in Distributed Environment
- Copy the train dataset to the same path of each machine, or simply place it to a shared file system, such as NFS
- Touch a host file to write each machine's IP address, such as ./hosts
- Invoke the application with MPI 

```
mpiexec -hostfile ./hosts -n 8 ./bin/huge_walk -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --make-undirected -o ./out/walks.txt -eoutput ./out/LJ-r_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5  -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 2
```

### Check the output files in "out" directory
