
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include <gemini/core/mpi.hpp>

int get_mpi_rank()
{
    int a;
    MPI_Comm_rank(MPI_COMM_WORLD, &a);
    return a;
}

int get_mpi_size()
{
    int a;
    MPI_Comm_size(MPI_COMM_WORLD, &a);
    return a;
}
