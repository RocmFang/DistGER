#include <cstring>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include "type.hpp"
#include "util.hpp"
#ifdef USE_MKL
#include "mkl.h"
#include <mutex>
#endif

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MPI_SCALAR MPI_FLOAT

FILE* flog = nullptr;

typedef float real;
typedef unsigned int uint;
typedef unsigned long long ulonglong;

ulonglong local_wih_visit = 0;
ulonglong other_wih_visit = 0;
ulonglong local_woh_visit = 0;
ulonglong other_woh_visit = 0;

WalkEngine<real_t, uint32_t> *graph;

double top_speed = 0.0;

struct vocab_word
{
    uint cn;
    char *word;
};

class sequence
{
public:
    int *indices;
    int *meta;
    int length;

    sequence(int len)
    {
        length = len;
        indices = (int *)_mm_malloc(length * sizeof(int), 64);
        meta = (int *)_mm_malloc(length * sizeof(int), 64);
    }
    ~sequence()
    {
        _mm_free(indices);
        _mm_free(meta);
    }
};

int binary = 0, debug_mode = 2;
bool disk = false;
int num_procs = 1, num_threads = 12, negative = 5, min_count = 5, min_reduce = 1, iter = 5, window = 5, batch_size = 11, my_rank = -1;
size_t hidden_size = 100;
size_t vocab_size = 0;
int vocab_max_size = 1000, min_sync_words = 1024, full_sync_times = 0;
int message_size = 1024; // MB
ulonglong train_words = 0, file_size = 0;
real alpha = 0.1f, sample = 1e-3f;
real model_sync_period = 0.1f;
const real EXP_RESOLUTION = EXP_TABLE_SIZE / (MAX_EXP * 2.0f);

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary
const int table_size = 1e8;

struct vocab_word *vocab = NULL;
int *vocab_hash = NULL;
int *table = NULL;
real *Wih = NULL, *Woh = NULL, *expTable = NULL;

vector<int> *cn_vocab;
vector<int> *local_corpus;
struct vocab_vertex
{
    int cn;
    vertex_id_t id;
    vocab_vertex(){};
    vocab_vertex(uint _cn, vertex_id_t _id) : cn(_cn), id(_id){};
};
vector<vocab_vertex> v_vocab;
vector<vertex_id_t> hash2v_vocab; 
vector<vertex_id_t> hash2new_sort;

vector<vertex_id_t> rehash2v_vocab;

vector<vertex_id_t> *new_sort;
vector<vertex_id_t> cn2new_sort;

double g_wih_access = 0.0;
double g_wih_read = 0.0;
double g_wih_update = 0.0;
double g_woh_accesss = 0.0;
double g_woh_read = 0.0;
double g_woh_update = 0.0;
double g_matrix_compute = 0.0;

mutex g_mutex;
char new_sort_file[MAX_STRING];
char save_raw_emb[MAX_STRING];
vector<int> word_freq_block_ind; 


void my_InitUnigramTable()
{
    table = (int *)_mm_malloc(table_size * sizeof(int), 64);

    const real power = 0.75f;
    double train_words_pow = 0.;
#pragma omp parallel for num_threads(num_threads) reduction(+ \
                                                            : train_words_pow)
    for (int i = 0; i < vocab_size; i++)
    {
        train_words_pow += pow(v_vocab[i].cn, power);
    }

    int i = 0;
    real d1 = pow(v_vocab[i].cn, power) / train_words_pow;
    for (int a = 0; a < table_size; a++)
    {
        table[a] = i; 
        if (a / (real)table_size > d1)
        {
            i++; 
            if (i >= vocab_size)
                i = vocab_size - 1;
            d1 += pow(v_vocab[i].cn, power) / train_words_pow;
        }
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin)
{
    int a = 0, ch;
    while (!feof(fin))
    {
        ch = fgetc(fin);
        if (ch == 13)
            continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (a > 0)
            {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n')
            {
                strcpy(word, (char *)"</s>");
                return;
            }
            else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1)
            a--; // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word)
{
    uint hash = 0;
    for (int i = 0; i < strlen(word); i++)
        hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word)
{
    int hash = GetWordHash(word);
    while (1)
    {
        if (vocab_hash[hash] == -1)
            return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin)
{
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin))
        return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word)
{
    int hash, length = strlen(word) + 1;
    if (length > MAX_STRING)
        length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size)
    {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b)
{
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void my_SortVocab()
{

    sort(v_vocab.begin(), v_vocab.end(), [](vocab_vertex a, vocab_vertex b)
         { return a.cn > b.cn; });

}

// Sorts the vocabulary by frequency using word counts
void SortVocab()
{
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    int size = vocab_size;
    train_words = 0;
    for (int i = 0; i < size; i++)
    {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[i].cn < min_count) && (i != 0))
        {
            vocab_size--;
            free(vocab[i].word);
        }
        else
        {
            // Hash will be re-computed, as after the sorting it is not actual
            int hash = GetWordHash(vocab[i].word);
            while (vocab_hash[hash] != -1)
                hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
            train_words += vocab[i].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, vocab_size * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab()
{
    int count = 0;
    for (int i = 0; i < vocab_size; i++)
    {
        if (vocab[i].cn > min_reduce)
        {
            vocab[count].cn = vocab[i].cn;
            vocab[count].word = vocab[i].word;
            count++;
        }
        else
        {
            free(vocab[i].word);
        }
    }
    vocab_size = count;
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    for (int i = 0; i < vocab_size; i++)
    {
        // Hash will be re-computed, as it is not actual
        int hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
    }
    min_reduce++;
}
void my_LearnVocabFromTrainFile()
{
    // FILE* f = fopen("./csr_cn.txt","w");
    vocab_size = cn_vocab->size();
    v_vocab.resize(vocab_size);
    for (int i = 0; i < v_vocab.size(); i++)
    {

        // v_vocab[i].cn = graph->vertex_out_degree[i];

        int degree = 0;
        int context_degree = 0;

    //    for (int p = graph->co_occor->adjList[i].begin; p < graph->co_occor->adjList[i].end; p++)
    //    {
    //        if (graph->co_occor->co_occor[p] > 0)
    //        {
    //            context_degree += graph->co_occor->co_occor[p];
    //            degree++;
    //        }
    //    }

    //    fprintf(f,"%d %d\n",context_degree,(*cn_vocab)[i]);

        // v_vocab[i].cn *= graph -> vertex_out_degree[i]; // 1

        // v_vocab[i].cn = degree;// 2

        // v_vocab[i].cn = context_degree;//3

         v_vocab[i].cn = (*cn_vocab)[i]; // 4

        v_vocab[i].id = i;
    }
    // fclose(f);
    for (int i = 0; i < cn_vocab->size(); i++)
    {
        train_words += (*cn_vocab)[i];
    }

    my_SortVocab();
    // printf("%d  sort done\n", my_rank);

   
    hash2new_sort.resize(vocab_size);
    hash2v_vocab.resize(vocab_size);
    rehash2v_vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++)
    {
        hash2v_vocab[v_vocab[i].id] = i;
        hash2new_sort[(*new_sort)[i]] = i;
    }
    for (int i = 0; i < vocab_size; i++)
    {
        rehash2v_vocab[hash2v_vocab[i]] = i;
    }
    assert(1000 == rehash2v_vocab[hash2v_vocab[1000]]);
    cn2new_sort.resize(vocab_size);
    for (int i = 0; i < cn2new_sort.size(); i++)
    {
        cn2new_sort[hash2v_vocab[i]] = hash2new_sort[i];
    }

    if (my_rank == 0 && debug_mode > 0)
    {
        printf("vertex Vocab size: %ld\n", vocab_size);
        printf("Words in global path %lld\n", train_words);
    }
}

void SaveVocab()
{
    FILE *fo = fopen(save_vocab_file, "wb");
    for (int i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab()
{
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL)
    {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    char c;
    vocab_size = 0;
    while (1)
    {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        int i = AddWordToVocab(word);
        auto read_count = fscanf(fin, "%d%c", &vocab[i].cn, &c);
    }
    SortVocab();
    if (debug_mode > 0 && my_rank == 0)
    {
        printf("Vocab size: %ld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fclose(fin);

    // get file size
    FILE *fin2 = fopen(train_file, "rb");
    if (fin2 == NULL)
    {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin2, 0, SEEK_END);
    file_size = ftell(fin2);
    fclose(fin2);
}

void InitNet()
{
    Wih = (real *)_mm_malloc(vocab_size * (hidden_size * sizeof(real)), 64);
    Woh = (real *)_mm_malloc(vocab_size * (hidden_size * sizeof(real)), 64);
    if (!Wih || !Woh)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

#pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    for (int i = 0; i < vocab_size; i++)
    {
        memset(Wih + i * hidden_size, 0.f, hidden_size * sizeof(real));
        memset(Woh + i * hidden_size, 0.f, hidden_size * sizeof(real));
    }

    // initialization
    ulonglong next_random = 1;
    for (int i = 0; i < vocab_size * hidden_size; i++)
    {
        next_random = next_random * (ulonglong)25214903917 + 11;
        Wih[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
    }
}

ulonglong loadStream(FILE *fin, int *stream, const ulonglong total_words)
{
    ulonglong word_count = 0;
    while (!feof(fin) && word_count < total_words)
    {
        int w = ReadWordIndex(fin);
        if (w == -1)
            continue;
        stream[word_count] = w;
        word_count++;
    }
    stream[word_count] = 0; 
    return word_count;
}
// assume v > 0
inline unsigned int getNumZeros(unsigned int v)
{
    unsigned int numzeros = 0;
    while (!(v & 0x1))
    {
        numzeros++;
        v = v >> 1;
    }
    return numzeros;
}

void convert_inputs(int *inputs, int input_num)
{
    for (int i = 0; i < input_num; i++)
    {
        inputs[i] = cn2new_sort[inputs[i]];
    }
}

void word_freq_block() // Ma
{

    word_freq_block_ind.push_back(0);
    int j = 1;
    for (int i = 1; i < v_vocab.size(); i++)
    {
        // printf("freq: %d %d \n",i, v_vocab[i].cn);
        if (v_vocab[i].cn != v_vocab[i - 1].cn)
        {
            word_freq_block_ind.push_back(i);
            // printf("freq: %d %d %d \n",i, j, v_vocab[i].cn);
            j++;
        }
    }
    printf("freq_block num: %d \n", j);
}

void preparatory_work()
{    
    my_LearnVocabFromTrainFile();
    printf("%d learn train file done\n", my_rank);
    InitNet();
    printf("%d InitNet done\n", my_rank);
    my_InitUnigramTable();
    word_freq_block();
}


void Train_SGNS_MPI()
{

#ifdef USE_MKL
    if (my_rank == 0)
        printf("=====use MKL=========\n");
    mkl_set_num_threads(1);
#endif

    int num_parts = num_procs * (num_threads - 1);

    int local_num_parts = num_threads - 1;

    real starting_alpha = alpha;
    ulonglong word_count_actual = 0;

    int ready_threads = 0;
    int active_threads = num_threads - 1;
    bool compute_go = true;

    

#pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();

        if (id == 0)
        {


            int active_processes = 1;
            int active_processes_global = num_procs;
            ulonglong word_count_actual_global = 0;
            int sync_chunk_size = message_size * 1024 * 1024 / (hidden_size * 4); 
                                                                               
            int full_sync_count = 1;
            unsigned int num_syncs = 0;
            while (ready_threads < num_threads - 1)
            {
                usleep(1);
            }
            MPI_Barrier(MPI_COMM_WORLD);
 
#pragma omp atomic
            ready_threads++;

            double start = omp_get_wtime(); 
            double sync_start = start;
            int sync_block_size = 0;

            while (1)
            {
                
            
                double sync_eclipsed = omp_get_wtime() - sync_start;
                // printf("sync_eclipsed:\n", sync_eclipsed);
                // model_sync_period = 0.1
                if (sync_eclipsed > model_sync_period) //model_sync_period
                
                {
                    // printf("sync_eclipsed>1\n");
                    
                    compute_go = false;
             
                    num_syncs++;
        
                    active_processes = (active_threads > 0 ? 1 : 0);

                    // synchronize parameters
                    
                    MPI_Allreduce(&active_processes, &active_processes_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&word_count_actual, &word_count_actual_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

                    // determine if full sync   getNumZero：获取有几个0
                    
                    // int sync_vocab_size = min((1 << getNumZeros(num_syncs)) * min_sync_words, vocab_size);
                    real progress = word_count_actual_global / (real)(iter * train_words + 1);

                     //============= all sync ================
                    
                    if ((full_sync_times > 0) && (progress > (real)full_sync_count / (full_sync_times + 1) + 0.01f))
                    {
                        // printf("p%d full sync No.%d \n",my_rank,full_sync_count);
                        full_sync_count++;
                        int sync_vocab_size = vocab_size; 
                        sync_block_size = vocab_size;

                        
                        int num_rounds = sync_vocab_size / sync_chunk_size + ((sync_vocab_size % sync_chunk_size > 0) ? 1 : 0);
                       
                        for (int r = 0; r < num_rounds; r++) 
                        {
                            int start = r * sync_chunk_size;
                            int sync_size = min(sync_chunk_size, sync_vocab_size - start);
                            

                            // printf("win+--> syn_size: %d  sync_vac_size: %d \n", sync_size, sync_vocab_size);
                            MPI_Allreduce(MPI_IN_PLACE, Wih + start * hidden_size, sync_size * hidden_size, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);
                            MPI_Allreduce(MPI_IN_PLACE, Woh + start * hidden_size, sync_size * hidden_size, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);
                        }
                       
    #pragma simd
    #pragma vector aligned
                        for (int i = 0; i <sync_vocab_size * hidden_size; i++)
                        {
                            Wih[i] /= num_procs; 
                            Woh[i] /= num_procs;
                        }
                    }
                    //========== all sync =============
                    else{

                        real *sync_block_in = (real *)_mm_malloc((size_t)vocab_size * hidden_size * sizeof(real), 64);
                        real *sync_block_out = (real *)_mm_malloc((size_t)vocab_size * hidden_size * sizeof(real), 64);

                        vector<int> sync_ind; //randomly generate the word in one context degree block

                        //////synchronize the random index between each computing nodes

                        if (my_rank == 0) // Ma
                        {
                            int i = 1;

                            for (int j = 1; j < word_freq_block_ind.size(); j += 1)
                            {

                                /////----> synchronizing one high frequency word in block and randomly select more than one word in low frequency words

                                // if (j>(word_freq_block_ind.size())*0.9)
                                // {

                                //     for (int t=0; t<2; t++)
                                //     {
                                //         int tem_ind = rand()%(word_freq_block_ind[j]-word_freq_block_ind[j-1])+word_freq_block_ind[j-1];
                                //         if (tem_ind != sync_ind[i-1])
                                //         {
                                //             sync_ind.push_back(tem_ind);
                                //             i++;
                                //             printf("cn: %d \n", v_vocab[tem_ind].cn);
                                //         }
                                //     }
                                // }
                                // else
                                // {
                                //     int tem_ind = rand()%(word_freq_block_ind[j]-word_freq_block_ind[j-1])+word_freq_block_ind[j-1];
                                //     sync_ind.push_back(tem_ind);
                                //         // printf("hig_blk: %d", tem_ind);
                                //     i++;
                                // }

                                // /////----> synchronizing all high frequency words and randomly select more than one word in low frequency words

                                // if (j>(word_freq_block_ind.size())*0.2)
                                // {

                                //     for (int t=0; t<3; t++)
                                //     {
                                //         int tem_ind = rand()%(word_freq_block_ind[j]-word_freq_block_ind[j-1])+word_freq_block_ind[j-1];
                                //         if (tem_ind != sync_ind[i-1])
                                //         {
                                //             sync_ind.push_back(tem_ind);
                                //             i++;
                                //         }
                                //     // int tem_ind = rand()%(word_freq_block_ind[j]-word_freq_block_ind[j-1])+word_freq_block_ind[j-1];
                                //     // sync_ind.push_back(tem_ind);
                                //     // // printf("low_blk: %d", tem_ind);
                                //     }
                                // }
                                // else
                                // {
                                //     for(int tem_ind=word_freq_block_ind[j-1]; tem_ind<word_freq_block_ind[j]; tem_ind++)
                                //     {
                                //         sync_ind.push_back(tem_ind);
                                //         // printf("hig_blk: %d", tem_ind);
                                //         i++;
                                //     }

                                // }

                                ///----> added one or more than one word in a block to synchronize
                                int i = 1;
                                for (int t = 0; t < 1; t++)
                                {
                                    int tem_ind = rand() % (word_freq_block_ind[j] - word_freq_block_ind[j - 1]) + word_freq_block_ind[j - 1];
                                    if (tem_ind == 0)
                                    {
                                        sync_ind.push_back(tem_ind);
                                    }
                                    else
                                    {
                                        if (tem_ind != sync_ind[i - 1])
                                        {
                                            sync_ind.push_back(tem_ind);
                                            i++;
                                        }
                                    }
                                }
                            }
                        }

                         sync_block_size = sync_ind.size();

                        MPI_Bcast(&sync_block_size, 1, get_mpi_data_type<int>(), 0, MPI_COMM_WORLD);
                        if (my_rank != 0)
                        {
                            sync_ind.resize(sync_block_size);
                        }
                        MPI_Bcast(sync_ind.data(), sync_block_size, get_mpi_data_type<int>(), 0, MPI_COMM_WORLD);

                        for (size_t i = 0; i < sync_block_size; i++)
                        {
                            // printf("id %d, sync_id %d\n",i, sync_ind[i]);

                            memcpy(sync_block_in + i * hidden_size, Wih + (size_t)sync_ind[i] * hidden_size, hidden_size * sizeof(real));
                            memcpy(sync_block_out + i * hidden_size, Woh + (size_t)sync_ind[i] * hidden_size, hidden_size * sizeof(real));
                        }

                        MPI_Allreduce(MPI_IN_PLACE, sync_block_in, sync_block_size * hidden_size, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);
                        MPI_Allreduce(MPI_IN_PLACE, sync_block_out, sync_block_size * hidden_size, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);

    #pragma simd
    #pragma vector aligned
                        for (size_t i = 0; i < sync_block_size * hidden_size; i++)
                        {
                            sync_block_in[i] /= num_procs; 
                            sync_block_out[i] /= num_procs;
                        }

                        for (size_t i = 0; i < sync_ind.size(); i++)
                        {
                            // printf("syn_ind: %d %d \n", i, sync_ind[i]);
                            memcpy(Wih + (size_t)sync_ind[i] * hidden_size, sync_block_in + (i)*hidden_size, hidden_size * sizeof(real));
                            memcpy(Woh + (size_t)sync_ind[i] * hidden_size, sync_block_out + (i)*hidden_size, hidden_size * sizeof(real));
                        }

                    }
                    // let it go!
                    compute_go = true;

                    // print out status
                    if (my_rank == 0 && debug_mode > 1)
                    {
                        double now = omp_get_wtime();
                        printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  Words sync'ed: %d", 13, alpha,
                               progress * 100,
                               word_count_actual_global / ((now - start) * 1000),
                               sync_block_size);
                        fflush(stdout);
                        if(word_count_actual_global / ((now - start) * 1000) > top_speed)top_speed = word_count_actual_global / ((now - start) * 1000) ;
                    }
                  
                    if (active_processes_global == 0)
                        break;
                    sync_start = omp_get_wtime();
                }
                else
                {
                    usleep(1); 
                }
            }
        }
        else
        {


            Timer timer, wiat, woat, modet, wop;

            
            int local_iter = iter;
            ulonglong next_random = my_rank * (num_threads - 1) + id - 1; // global 计算计算线程 id ，
            ulonglong word_count = 0, last_word_count = 0;
            int sentence_length = 0, sentence_position = 0;
           
            int sen[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

            int sentence_length2 = 0, sentence_position2 = 0;
            int sen2[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

            int train_sen_num = 0;
            bool change_neg = true;
            int neg_count = 0;
        

            FILE *fin = fopen(train_file, "rb");
            // fseek(fin, file_size * (my_rank * (num_threads - 1) + id - 1) / num_parts, SEEK_SET);
            // ulonglong local_train_words = train_words / num_parts + (train_words % num_parts > 0 ? 1 : 0);

            

            ulonglong words_num = (*local_corpus).size();

            // ulonglong thread_words_num = words_num / local_num_parts + (words_num % local_num_parts > 0 ? 1 : 0);
            ulonglong thread_words_num = words_num / local_num_parts ;

            size_t start_words = thread_words_num * (id - 1);

            if (words_num % local_num_parts > 0 && id == (num_threads - 1))
            {
                thread_words_num = words_num - (local_num_parts - 1) * thread_words_num;
            }
            ulonglong real_word = 0;

            // printf("id %d  part ready\n",id);

            size_t cur_words = start_words;

            int w;

            int *stream;

            batch_size = 2 * window + 1;
            // temporary memory
            real *inputM = (real *)_mm_malloc(2 * batch_size * hidden_size * sizeof(real), 64);
            real *outputM = (real *)_mm_malloc((2 + negative) * hidden_size * sizeof(real), 64);
            real *outputMd = (real *)_mm_malloc((2 + negative) * hidden_size * sizeof(real), 64);
            real *corrM = (real *)_mm_malloc((2 + negative) * 2 * batch_size * sizeof(real), 64);

            // ========== first sentence input buffer and target buffer ====================
            int INPUT_BUFFER_MAX_SIZE = 22;
            real *inputBuffer = (real *)_mm_malloc(INPUT_BUFFER_MAX_SIZE * hidden_size * sizeof(real), 64);
            // real *targetBuffer = (real *)_mm_malloc(INPUT_BUFFER_MAX_SIZE * hidden_size * sizeof(real), 64);
            vector<int> sen_words; // the variety nums of words in one sentence

            // ========== second sentence input buffer and target buffer =====================
            int INPUT_BUFFER2_MAX_SIZE = 22;
            real *inputBuffer2 = (real *)_mm_malloc(INPUT_BUFFER2_MAX_SIZE * hidden_size * sizeof(real), 64);
            // real *targetBuffer2 = (real *)_mm_malloc(INPUT_BUFFER_MAX_SIZE * hidden_size * sizeof(real), 64);
            vector<int> sen_words2;

            // ========== common negative buffer =======================
            int NEGATIVE_BUFFER_MAX_SIZE = 100;
            real *negativeBuffer = (real *)_mm_malloc(NEGATIVE_BUFFER_MAX_SIZE * hidden_size * sizeof(real), 64);
            vector<int> neg_buffer;
            vector<int> neg_meta;
            vector<int> neg_words;

            // int sen_words_num;
            int neg_batch_num;

            // int inputs[2 * window + 1] __attribute__((aligned(64)));
            int inputs[2 * (2 * window + 1)] __attribute__((aligned(64)));

            // sequence outputs(1 + negative);
            sequence outputs(2 + negative);

            
#pragma omp atomic
            ready_threads++;
            
            while (ready_threads < num_threads)
            {
                usleep(1);
            }
            // printf("p%d t%d alloc\n",my_rank,id);
            while (1)
            { // 计算时间
                modet.restart();
                while (!compute_go)
                {
                    usleep(1);
                }
                modet.restart();

                
                if (word_count - last_word_count > 10000)
                {
                    ulonglong diff = word_count - last_word_count;
#pragma omp atomic
                    word_count_actual += diff;
                    last_word_count = word_count;

                    // update alpha
                    alpha = starting_alpha * (1 - word_count_actual * num_procs / (real)(iter * train_words + 1));
                    if (alpha < starting_alpha * 0.0001f)
                        alpha = starting_alpha * 0.0001f;
                }

               
                // 把词读到句子中
                // if (sentence_length == 0)
                while(sentence_length == 0)
                {
                     if (cur_words >= start_words + thread_words_num|| cur_words>=words_num)
                    {
                        break;
                    }
                    // for(int i=0;i<(*local_path)[cur_words].size();i++)
                    while (true)
                    {

                        int origin_id = (*local_corpus)[cur_words];
                        word_count++;
                        cur_words++;
                        if (origin_id == -1 || cur_words >= start_words + thread_words_num)
                            break;
                        w = hash2v_vocab[origin_id]; 
                        // w= hash2new_sort[origin_id];

                        // The subsampling randomly discards frequent words while keeping the ranking same
                        if (sample > 0)
                        {
                            real ratio = (sample * train_words) / (*cn_vocab)[w];
                            real ran = sqrtf(ratio) + ratio;
                            next_random = next_random * (ulonglong)25214903917 + 11;
                            if (ran < (next_random & 0xFFFF) / 65536.f) 
                                continue;
                        }
                        sen[sentence_length] = w;
                        sentence_length++;
                        if (sentence_length >= MAX_SENTENCE_LENGTH)
                            break;
                    }
                    sentence_position = 0;
                    if (sentence_length == 1|| sentence_length==0)
                    {
                        sentence_length = 0;
                        continue;
                    }
                    sen_words.clear();
		    timer.restart();
                    for (int i = 0; i < sentence_length; i++)
                    {
                        int word = sen[i];
                        auto pos = find(sen_words.begin(), sen_words.end(), word);
                        if (pos == sen_words.end()) // if there isn't w in sen_words
                        {
                            sen_words.push_back(word);
                        }
                    }

                    if (INPUT_BUFFER_MAX_SIZE < sen_words.size())
                    {
                        _mm_free(inputBuffer);
                        INPUT_BUFFER_MAX_SIZE = sen_words.size();
                        inputBuffer = (real *)_mm_malloc(((size_t)INPUT_BUFFER_MAX_SIZE)*hidden_size * sizeof(real), 64);
                    }

                    for (size_t i = 0; i < sen_words.size(); i++)
                    {
                        memcpy(inputBuffer + i * hidden_size, Wih + (size_t)sen_words[i] * hidden_size, hidden_size * sizeof(real));
                    }
                    // fprintf(flog,"p%d t%d load sen1 len: %d\n",my_rank,id,sentence_length);
                    // fflush(flog);
                }
                 
                // if (sentence_length2 == 0)
                while(sentence_length2 == 0)
                {
                    if (cur_words >= start_words + thread_words_num|| cur_words>=words_num)
                    {
                        break;
                    }
                    while (true)
                    {

                        int origin_id = (*local_corpus)[cur_words];
                        word_count++;
                        cur_words++;
                        if (origin_id == -1 || cur_words >= start_words + thread_words_num)
                            break;
                        w = hash2v_vocab[origin_id]; 
                        // w= hash2new_sort[origin_id];

                        // The subsampling randomly discards frequent words while keeping the ranking same
                        if (sample > 0)
                        {
                            real ratio = (sample * train_words) / (*cn_vocab)[w];
                            real ran = sqrtf(ratio) + ratio;
                            next_random = next_random * (ulonglong)25214903917 + 11;
                            if (ran < (next_random & 0xFFFF) / 65536.f) // 拒绝，重新采样
                                continue;
                        }
                        sen2[sentence_length2] = w;
                        sentence_length2++;
                        if (sentence_length2 >= MAX_SENTENCE_LENGTH)
                            break;
                    }
                    sentence_position2 = 0;
                    if (sentence_length2 == 1 || sentence_length2 == 0)
                    {
                        sentence_length2 = 0;
                        continue;
                    }
                    sen_words2.clear();
		    timer.restart();
                    for (int i = 0; i < sentence_length2; i++)
                    {
                        int word = sen2[i];
                        auto pos = find(sen_words2.begin(), sen_words2.end(), word);
                        if (pos == sen_words2.end()) // if there isn't w in sen_words
                        {
                            sen_words2.push_back(word);
                        }
                    }

                    if (INPUT_BUFFER2_MAX_SIZE < sen_words2.size())
                    {
                        _mm_free(inputBuffer2);
                        INPUT_BUFFER2_MAX_SIZE = sen_words2.size();
                        inputBuffer2 = (real *)_mm_malloc(((size_t)INPUT_BUFFER2_MAX_SIZE)*hidden_size * sizeof(real), 64);
                    }

                    for (size_t i = 0; i < sen_words2.size(); i++)
                    {
                        memcpy(inputBuffer2 + i * hidden_size, Wih + (size_t)sen_words2[i] * hidden_size, hidden_size * sizeof(real));
                    }
                // fprintf(flog,"p%d t%d load sen2 len: %d\n",my_rank,id,sentence_length);
                // fflush(flog);
                }
                  if (cur_words >= start_words + thread_words_num|| cur_words>=words_num)
                {
                    ulonglong diff = word_count - last_word_count;
#pragma omp atomic
                    word_count_actual += diff;

                    local_iter--;
                    if (local_iter == 0)
                    {
#pragma omp atomic
                        active_threads--;
                        break;
                    }
                    word_count = 0;
                    cur_words = start_words;
                    last_word_count = 0;
                    sentence_length = 0;
                    sentence_length2 = 0;
                    continue;
                }
                // fprintf(flog,"p%d t%d cur word idx %zu  vid: %zu start word idx: %zu vid %zu \n",my_rank,id,cur_words,(*local_corpus)[cur_words],start_words,(*local_corpus)[start_words]);
                // fflush(flog);
                // ================ neg words ===================
                if (change_neg)
                {
                    neg_buffer.clear();
                    neg_words.clear();
                    neg_batch_num = max(sentence_length, sentence_length2);
                    //neg_batch_num = 5;
                    for (int i = 0; i < neg_batch_num; i++)
                    {
                        for (int k = 0; k < negative; k++)
                        {
                            next_random = next_random * (ulonglong)25214903917 + 11;
                            int sample = table[(next_random >> 16) % table_size];
                            if (!sample)
                                sample = next_random % (vocab_size - 1) + 1;
                            neg_words.push_back(sample);
                            auto p = find(neg_buffer.begin(), neg_buffer.end(), sample);
                            if (p == neg_buffer.end())
                                neg_buffer.push_back(sample);
                        }
                    }
                    if (NEGATIVE_BUFFER_MAX_SIZE < neg_buffer.size())
                    {
                        _mm_free(negativeBuffer);
                        NEGATIVE_BUFFER_MAX_SIZE = neg_buffer.size();
                        negativeBuffer = (real *)_mm_malloc(((size_t)NEGATIVE_BUFFER_MAX_SIZE)*hidden_size * sizeof(real), 64);
                    }

                    for (size_t i = 0; i < neg_buffer.size(); i++)
                    {
                        memcpy(negativeBuffer + i * hidden_size, Woh + (size_t)neg_buffer[i] * hidden_size, hidden_size * sizeof(real));
                    }
                    change_neg = false;
                //  fprintf(flog,"p%d t%d change neg\n",my_rank,id);
                //  fflush(flog);
                }
                int target = sen[sentence_position];
                int target2 = sen2[sentence_position2];
                // fprintf(flog,"p%d t%d sen pos: %d target1 %d sen pos2: %d target2 %d\n",my_rank,id,sentence_position,target,sentence_position2,target2);
                // fflush(flog);
                assert(target != -1);

                real_word += 2;
                outputs.indices[0] = target;
                outputs.meta[0] = 1;

                // get all input contexts around the target word
                next_random = next_random * (ulonglong)25214903917 + 11;
                int b = next_random % window;

                int num_inputs = 0; 
            
                for (int i = b; i < 2 * window + 1 - b; i++)
                {
                    if (i != window)
                    {
                        int c = sentence_position - window + i; 
                        if (c < 0)
                            continue;
                        if (c >= sentence_length)
                            break;
                        inputs[num_inputs] = sen[c];
                        num_inputs++;
                    }
                }
                int first_input_num = num_inputs;

                for (int i = b; i < 2 * window + 1 - b; i++)
                {
                    if (i != window)
                    {
                        int c = sentence_position2 - window + i;
                        if (c < 0)
                            continue;
                        if (c >= sentence_length2)
                            break;
                        inputs[num_inputs] = sen2[c];
                        num_inputs++;
                    }
                }
                //  fprintf(flog,"p%d t%d gen inputs\n",my_rank,id);
                //  fflush(flog);
                // ======================== select negative =======================
                neg_count++;
                int start = neg_count % neg_batch_num;
                int offset = 1;
                for (int k = 0; k < negative; k++)
                {
                    int sample = neg_words[start * negative + k];
                    int *p = find(outputs.indices, outputs.indices + offset, sample);
                    if (p == outputs.indices + offset)
                    {
                        outputs.indices[offset] = sample;
                        outputs.meta[offset] = 1;
                        offset++;
                    }
                    else
                    {
                        int idx = p - outputs.indices;
                        outputs.meta[idx]++;
                    }
                }
                outputs.indices[0] = target;
                outputs.meta[0] = 1;
                outputs.indices[offset] = target2;
                outputs.meta[offset] = 1;
                outputs.length = offset + 1;
                //  fprintf(flog,"p%d t%d gen neg\n",my_rank,id);
                //  fflush(flog);
                // outputs.length = negative + 1;

                // fetch input sub model
                // int input_start = b * batch_size;
                // int input_size = min(batch_size, num_inputs - input_start);
                int input_size = num_inputs;
                // assert(input_size>0);
                // for (int i = 0; i < input_size; i++) {
                //     memcpy(inputM + i * hidden_size, Wih + inputs[input_start + i] * hidden_size, hidden_size * sizeof(real));
                // }
                //====================  copy input from input buf =============================
                vector<size_t> input_buf_offset(input_size);
                for (int i = 0; i < first_input_num; i++)
                {
                    int idx = find(sen_words.begin(), sen_words.end(), inputs[i]) - sen_words.begin();
                    input_buf_offset[i] = idx * hidden_size;
                    memcpy(inputM + i * hidden_size, inputBuffer + input_buf_offset[i], hidden_size * sizeof(real));
                }
                for (int i = first_input_num; i < input_size; i++)
                {
                    int idx = find(sen_words2.begin(), sen_words2.end(), inputs[i]) - sen_words2.begin();
                    input_buf_offset[i] = idx * hidden_size;
                    memcpy(inputM + i * hidden_size, inputBuffer2 + input_buf_offset[i], hidden_size * sizeof(real));
                }
                //  fprintf(flog,"p%d t%d get input vector\n",my_rank,id);
                //  fflush(flog);
                // accessWih(false,input_size,input_start,inputs,inputM);

                // fetch output sub model
                int output_size = outputs.length;
                // for (int i = 0; i < output_size; i++) {
                //     memcpy(outputM + i * hidden_size, Woh + outputs.indices[i] * hidden_size, hidden_size * sizeof(real));
                // }
                // printf("copy input buf\n");
                // ========= copy target from target buffer ==============
                vector<int> neg_buf_offset(output_size - 2);
                // int target_idx = find(sen_words.begin(), sen_words.begin(), target) - sen_words.begin();
                // int target_idx2 = find(sen_words2.begin(), sen_words2.begin(), target2) - sen_words2.begin();
                // neg_buf_offset[0] = target * hidden_size;
                // neg_buf_offset[output_size-1] = target2 * hidden_size;
                memcpy(outputM, Woh + (size_t)target * hidden_size, hidden_size * sizeof(real));
                memcpy(outputM + (output_size - 1) * hidden_size, Woh + (size_t)target2 * hidden_size, hidden_size * sizeof(real));
                // printf("copy target buf\n");
                // ======== copy negative from negative buffer ===============
                for (int i = 0; i < output_size - 2; i++)
                {
                    int p = find(neg_buffer.begin(), neg_buffer.end(), outputs.indices[i + 1]) - neg_buffer.begin();
                    neg_buf_offset[i] = p * hidden_size;
                    memcpy(outputM + (i + 1) * hidden_size, negativeBuffer + neg_buf_offset[i], hidden_size * sizeof(real));
                }
                // accessWoh(false,output_size,outputs,outputM);

                // printf("copy negative buf\n");
		        timer.restart();
                //  fprintf(flog,"p%d t%d get output vector\n",my_rank,id);
                //  fflush(flog);
#ifndef USE_MKL
                for (int i = 0; i < output_size; i++)
                {
                    int c = outputs.meta[i];
                    for (int j = 0; j < input_size; j++)
                    {
                        real f = 0.f, g;
#pragma simd
                        for (int k = 0; k < hidden_size; k++)
                        {
                            f += outputM[i * hidden_size + k] * inputM[j * hidden_size + k];
                        }
                        int label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            g = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            g = label * alpha;
                        else
                            g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        corrM[i * input_size + j] = g * c;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, input_size, hidden_size, 1.0f, outputM,
                            hidden_size, inputM, hidden_size, 0.0f, corrM, input_size);
                for (int i = 0; i < output_size; i++)
                {
                    int c = outputs.meta[i];
                    int offset = i * input_size;
#pragma simd
                    for (int j = 0; j < first_input_num; j++)
                    {
                        real f = corrM[offset + j];
                        int label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            f = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            f = label * alpha;
                        else
                            f = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        corrM[offset + j] = f * c;
                    }
                    for (int j = first_input_num; j < input_size; j++)
                    {
                        real f = corrM[offset + j];
                        int label = (i != output_size - 1 ? 0 : 1);
                        if (f > MAX_EXP)
                            f = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            f = label * alpha;
                        else
                            f = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        corrM[offset + j] = f * c;
                    }
                }
#endif

#ifndef USE_MKL
                for (int i = 0; i < output_size; i++)
                {
                    for (int j = 0; j < hidden_size; j++)
                    {
                        real f = 0.f;
#pragma simd
                        for (int k = 0; k < input_size; k++)
                        {
                            f += corrM[i * input_size + k] * inputM[k * hidden_size + j];
                        }
                        outputMd[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, hidden_size, input_size, 1.0f, corrM,
                            input_size, inputM, hidden_size, 0.0f, outputMd, hidden_size);
#endif

#ifndef USE_MKL
                for (int i = 0; i < input_size; i++)
                {
                    for (int j = 0; j < hidden_size; j++)
                    {
                        real f = 0.f;
#pragma simd
                        for (int k = 0; k < output_size; k++)
                        {
                            f += corrM[k * input_size + i] * outputM[k * hidden_size + j];
                        }
                        inputM[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, input_size, hidden_size, output_size, 1.0f, corrM,
                            input_size, outputM, hidden_size, 0.0f, inputM, hidden_size);
#endif


                for (int i = 0; i < first_input_num; i++)
                {
                    int src = i * hidden_size;
                    int des = input_buf_offset[i];
#pragma simd
                    for (int j = 0; j < hidden_size; j++)
                    {
                        inputBuffer[des + j] += inputM[src + j];
                    }
                }
                for (int i = first_input_num; i < input_size; i++)
                {
                    int src = i * hidden_size;
                    int des = input_buf_offset[i];
#pragma simd
                    for (int j = 0; j < hidden_size; j++)
                    {
                        inputBuffer2[des + j] += inputM[src + j];
                    }
                }
                // fprintf(flog,"p%d t%d update input buffer \n",my_rank,id);
                //  fflush(flog);
                // accessWih(true,input_size,input_start,inputs,inputM);

                // for (int i = 0; i < output_size; i++) {
                //     int src = i * hidden_size;
                //     int des = outputs.indices[i] * hidden_size;
                //     #pragma simd
                //     for (int j = 0; j < hidden_size; j++) {
                //         Woh[des + j] += outputMd[src + j];
                //     }
                // }
                size_t src1 = 0;
                size_t des1 = (size_t)target * hidden_size;

                size_t src2 = (size_t)(output_size - 1) * hidden_size;
                size_t des2 = (size_t)target2 * hidden_size;

                for (int j = 0; j < hidden_size; j++)
                {
                    Woh[des1 + j] += outputMd[src1 + j];
                    Woh[des2 + j] += outputMd[src2 + j];
                }
                // fprintf(flog,"p%d t%d update  target buffer \n",my_rank,id);
                //  fflush(flog);
                for (int i = 0; i < neg_buf_offset.size(); i++)
                {
                    int src = (i + 1) * hidden_size;
                    int des = neg_buf_offset[i];
#pragma simd
                    for (int j = 0; j < hidden_size; j++)
                    {
                        negativeBuffer[des + j] += outputMd[src + j];
                    }
                }
                // accessWoh(true,output_size,outputs,outputMd);
                // }

                sentence_position++;
                sentence_position2++;
                //  fprintf(flog,"p%d t%d update  woh buffer \n",my_rank,id);
                //  fflush(flog);
                // printf("update buf\n");
                if (sentence_position >= sentence_length)
                {
                    // sync buffer to the wih and woh
                    for (size_t i = 0; i < sen_words.size(); i++)
                    {
                        memcpy(Wih + (size_t)sen_words[i] * hidden_size, inputBuffer + i * hidden_size, hidden_size * sizeof(real));
                        // memcpy(Woh + sen_words[i] * hidden_size, targetBuffer + i * hidden_size, hidden_size * sizeof(real));
                    }
                    sentence_length = 0;
                    train_sen_num++;
                }
                if (sentence_position2 >= sentence_length2)
                {
                    // sync buffer to the wih and woh
                    for (size_t i = 0; i < sen_words2.size(); i++)
                    {
                        memcpy(Wih + (size_t)sen_words2[i] * hidden_size, inputBuffer2 + i * hidden_size, hidden_size * sizeof(real));
                        // memcpy(Woh + sen_words2[i] * hidden_size, targetBuffer2 + i * hidden_size, hidden_size * sizeof(real));
                    }
                    sentence_length2 = 0;
                    train_sen_num++;
                }
                if (train_sen_num >= 2)
                {
                    for (size_t i = 0; i < neg_buffer.size(); i++)
                    {
                        memcpy(Woh + (size_t)neg_buffer[i] * hidden_size, negativeBuffer + i * hidden_size, hidden_size * sizeof(real));
                    }
                    change_neg = true;
                    train_sen_num = 0;
                }
            }
            _mm_free(inputM);
            _mm_free(outputM);
            _mm_free(outputMd);
            _mm_free(corrM);
            _mm_free(inputBuffer);
            _mm_free(inputBuffer2);
            _mm_free(negativeBuffer);
            if (disk)
            {
                fclose(fin);
            }
            else
            {
                _mm_free(stream);
            }

        }
    }
}

int ArgPos(char *str, int argc, char **argv)
{
    for (int a = 1; a < argc; a++)
        if (!strcmp(str, argv[a]))
        {
            // if (a == argc - 1) {
            //     printf("Argument missing for %s\n", str);
            //     exit(1);
            // }
            return a;
        }
    return -1;
}

void my_saveModel()
{
    // save the model
    FILE *fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%ld %ld\n", vocab_size, hidden_size);
    for (int a = 0; a < vocab_size; a++)
    {
        fprintf(fo, "%u ", v_vocab[a].id);
        if (binary)
            for (int b = 0; b < hidden_size; b++)
                fwrite(&Wih[a * hidden_size + b], sizeof(real), 1, fo);
        else
            for (int b = 0; b < hidden_size; b++)
                fprintf(fo, "%f ", Wih[a * hidden_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}


int dsgl(int argc, char **argv, vector<int> *vertex_cn, vector<vertex_id_t> *_new_sort, WalkEngine<real_t, uint32_t> *_graph)
{

    cn_vocab = vertex_cn;
    new_sort = _new_sort;
    graph = _graph;
    local_corpus = &graph->local_corpus;

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Get_processor_name(hostname, &hostname_len);

    printf("rank:%d  vertex cn %lu cn vocab %lu  local corpus size %lu \n", my_rank, vertex_cn->size(), cn_vocab->size(), local_corpus->size());

    // std::string log_path = "./log/" + std::string(hostname)+ ".log";
    // flog = fopen(log_path.c_str(),"w");

    printf("processor name: %s, number of processors: %d, rank: %d\n", hostname, num_procs, my_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    new_sort_file[0] = 0;
    save_raw_emb[0] = 0;

    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
        hidden_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
        strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
        strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0)
        debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
        binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
        alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-eoutput", argc, argv)) > 0)
        strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
        window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
        negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
        num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
        iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-batch-size", argc, argv)) > 0)
        batch_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-disk", argc, argv)) > 0)
        disk = true;
    if ((i = ArgPos((char *)"-sync-period", argc, argv)) > 0)
        model_sync_period = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-sync-words", argc, argv)) > 0)
        min_sync_words = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-full-sync-times", argc, argv)) > 0)
        full_sync_times = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-message-size", argc, argv)) > 0)
        message_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-save-emb", argc, argv)) > 0)
        strcpy(save_raw_emb, argv[i + 1]);


    if (my_rank == 0)
    {
        printf("number of processors: %d\n", num_procs);
        printf("number of threads: %d\n", num_threads);
        printf("number of iterations: %d\n", iter);
        printf("hidden size: %ld\n", hidden_size);
        printf("number of negative samples: %d\n", negative);
        printf("window size: %d\n", window);
        printf("batch size: %d\n", batch_size);
        printf("starting learning rate: %.5f\n", alpha);
        printf("stream from disk: %d\n", disk);
        printf("model sync period (secs): %.5f\n", model_sync_period);
        printf("minimal words sync'ed each time: %d\n", min_sync_words);
        printf("full model sync-up times: %d\n", full_sync_times);
        printf("MPI message chunk size (MB): %d\n", message_size);
        printf("starting training using file: %s\n\n", train_file);
    }
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)_mm_malloc(vocab_hash_size * sizeof(int), 64);
    expTable = (real *)_mm_malloc((EXP_TABLE_SIZE + 1) * sizeof(real), 64);
    for (i = 0; i < EXP_TABLE_SIZE + 1; i++)
    {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    Timer pw_timer;
    preparatory_work();
    // printf("\n> [preparatory_work TIME:] %lf \n", pw_timer.duration());


    // word_freq_block();

    Timer sgns_timer;
    Train_SGNS_MPI();
    printf("> [p%d Train SGNS MPI TIME:] %lf \n",my_rank, sgns_timer.duration());

    
    if (my_rank == 0)
    {
        printf("[ Top Speed ] %.2f k\n",top_speed); 
        my_saveModel();
    }
    // fclose(flog);

    return 0;
}
