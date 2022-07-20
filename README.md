<meta name="robots" content="noindex">

# Prerequisites
- Ubuntu 16.04
- Linux kernel 4.15.0
- g++ 9.4.0
- cmake 3.10.2
- [MPICH 3.4.2](https://www.mpich.org/)
- [MKL 2022.0.2](https://software.intel.com/en-us/mkl)

# Datasets
The evaluated dataset Youtube and LiveJournal are prepraed in the "dataset" directory.
Since the the space limited of the repository, the other dataset [Twitter](https://law.di.unimi.it/datasets.php), [Com-Orkut](https://snap.stanford.edu/) and [Flickr](http://datasets.syr.edu/pages/datasets.html) can be found in its open resource.


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
If we need to run the train data for the downstream tasks, such as Link prediction, we also should to relable the test data.

```
./bin/mpgp [train_data]  [test_data]  [vertex_num]  [partition_num]
```

```
cd build

./bin/mpgp ../dataset/LiveJournal.train  ../dataset/LiveJournal.test 2238731 8
```

The partitioned dataset will be saved in the input dataset directory. 

# Graph embedding

To start the embedding, we fist need to cover the train graph to binary format

```
./bin/gconverter -i ../dataset/LJ.train-8-r -o ./LJ-8.data-r -s weighted
```

Then create the "out" directory to save the walks file or embedding file

```
mkdir out
```

### Run in single-machine Environment
```
mpiexec -n 8 ./bin/huge_walk -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --min_L 20 --min_R 5 --make-undirected -o ./out/walks.txt -eoutput ./out/LJ-r_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5  -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 
```

### Run in Distributed Environment
- Copy the train dataset to the same path of each machine, or simply place it to a shared file system, such as NFS
- Touch a host file to write each machine's IP address, such as ./hosts
- Invoke the application with MPI 

```
mpiexec -hostfile ./hosts -n 8 ./bin/huge_walk -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --make-undirected -o ./out/walks.txt -eoutput ./out/LJ-r_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5  -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 
```

### Check the output files in "out" directory
