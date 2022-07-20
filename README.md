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
cd build

./bin/mpgp ../dataset/LiveJournal.train  ../dataset/LiveJournal.test 2238731 8
```

# Graph embedding
