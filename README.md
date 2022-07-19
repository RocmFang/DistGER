<meta name="robots" content="noindex">

# Environment
- Ubuntu 16.04
- inux kernel 4.15.0
- [MPICH 3.4](https://www.mpich.org/)
- [MKL 2022.0.2](https://software.intel.com/en-us/mkl)

# Dataset
The evaliated dataset Youtube and LiveJournal are prepraed in the "dataset" directory.
Since the the space limited of the repository, the other dataset [Twitter](https://law.di.unimi.it/datasets.php), [Com-Orkut](https://snap.stanford.edu/) and [Flickr](http://datasets.syr.edu/pages/datasets.html) can be found in its open resource.


# Setup
First Compile DistGER with CMake:

```
mkdir build && cd build

cmake ..

make
```

Then the compiled executable files are installed at the "bin" directory:

```
ls ./bin
```

# Partitioning


# Graphembedding
