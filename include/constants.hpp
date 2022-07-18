

#pragma once

#define L1_CACHE_LINE_SIZE 64

#ifdef UNIT_TEST

#define THREAD_LOCAL_BUF_CAPACITY 16
#define OMP_PARALLEL_THRESHOLD 10
#define PARALLEL_CHUNK_SIZE 4
#define PHASED_EXECTION_THRESHOLD 100
#define DISTRIBUTEDEXECUTIONCTX_PHASENUM 5
#define FOOT_PRINT_CHUNK_SIZE 16

#else

#define THREAD_LOCAL_BUF_CAPACITY 1024
#define OMP_PARALLEL_THRESHOLD 4000
#define PARALLEL_CHUNK_SIZE 128
#define PHASED_EXECTION_THRESHOLD 500000
#define DISTRIBUTEDEXECUTIONCTX_PHASENUM 16
#define FOOT_PRINT_CHUNK_SIZE 65536

#endif
