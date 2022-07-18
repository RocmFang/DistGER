
#pragma once

#include <string.h>
#include <vector>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "type.hpp"
#include "util.hpp"
#include "constants.hpp"
#include "mpi_helper.hpp"

class CoOccorCsr;


struct Footprint
{
    walker_id_t walker;
    vertex_id_t vertex;
    step_t step;
    Footprint () {}
    Footprint (walker_id_t _walker, vertex_id_t _vertex, step_t _step) : walker(_walker), vertex(_vertex), step(_step) {}
};

struct PathSet
{
    // path是分布式存储的，每个KnightKing实例holds paths的一个部分。对于每一个实例，path用几个segments来存储
    // 但是保证相同walker的路径存储在连续的空间
    // the seg_num表明有多少段
    int seg_num;                         // 对应的是23
    // vertex_id_t **path_set;
    // 第i段第j个路径的walker id
    walker_id_t **walker_id;
    // 第i段第j条路径的开始位置和结束位置
    vertex_id_t ***path_begin;
    vertex_id_t ***path_end;
    // 第i段第j条路径的长度
    step_t **path_length;
    // 第i段有多少个路径
    walker_id_t *path_num;
    PathSet ()
    {
        seg_num = 0;
        // path_set = nullptr;
        walker_id = nullptr;
        path_begin = nullptr;
        path_end = nullptr;
        path_length = nullptr;
        path_num = nullptr;
    }
    ~PathSet ()
    {
        if (seg_num != 0)
        {
            for (int s_i = 0; s_i < seg_num; s_i++)
            {
                // delete []path_set[s_i];
                delete []walker_id[s_i];
                delete []path_begin[s_i];
                delete []path_end[s_i];
                delete []path_length[s_i];
            }
            // delete []path_set;
            delete []walker_id;
            delete []path_begin;
            delete []path_end;
            delete []path_length;
            delete []path_num;
        }
    }
    // 将游走的数据写入到文件中
    void dump(const char* output_path, const char* fopen_mode, bool with_head_info ,std::vector<vertex_id_t> &vec,std::vector<int>&local_corpus,std::vector<int>&vertex_cn,CoOccorCsr* cocsr)
    {
        Timer timer;
        FILE* f = fopen(output_path, fopen_mode);
        assert(f != NULL);
        
        size_t null_sen = 0;
        // 每个线程负责的walker
        for (int worker_idx = 0; worker_idx < seg_num; worker_idx++)
        {
            // 第i段的所有路径
            for (walker_id_t walker_idx = 0; walker_idx < path_num[worker_idx]; walker_idx++)
            {   
                // std::vector<vertex_id_t>tmp_path;// 添加一条路径
                // with_head_info = true;
                // std::cout << "with_head_info = " << with_head_info << "\n";

                if (with_head_info)
                {
                    // 在写数据的时候写入walker_id和walker的长度
                    // fprintf(f, "%u %u", walker_id[worker_idx][walker_idx], path_length[worker_idx][walker_idx]);
                }
                if(path_length[worker_idx][walker_idx]==0)null_sen++;
                for (step_t p_i = 0; p_i < path_length[worker_idx][walker_idx]; p_i++)
                {
                    // 第i段第j个walker的地址的值
                    vec[*(path_begin[worker_idx][walker_idx] + p_i)]++;
                    // fprintf(f, " %u", *(path_begin[worker_idx][walker_idx] + p_i));
                    // tmp_path.push_back(*(path_begin[worker_idx][walker_idx] + p_i));

                    vertex_cn[*(path_begin[worker_idx][walker_idx] + p_i)]++;
                    local_corpus.push_back(*(path_begin[worker_idx][walker_idx] + p_i));
                    assert(*(path_begin[worker_idx][walker_idx] + p_i)<=INT_MAX);

                    // if(p_i != path_length[worker_idx][walker_idx]-1){
                    //     vertex_id_t cur = *(path_begin[worker_idx][walker_idx] + p_i);
                    //     vertex_id_t next = *(path_begin[worker_idx][walker_idx] + p_i + 1);
                    //     cocsr->add_co_occor(cur,next);
                    // }else{
                    //     vertex_id_t cur = *(path_begin[worker_idx][walker_idx] + p_i);
                    //      cocsr->add_co_occor(cur,cur);
                    // }
                }
                fprintf(f, "\n");
                // local_walk_res.push_back(tmp_path);
                local_corpus.push_back(-1);
            }
        }
        fclose(f);
        printf("p%d null sen: %zu\n",get_mpi_rank(),null_sen);
#ifndef UNIT_TEST
        printf("[p%d] finish write path data in %lf seconds \n",get_mpi_rank(), timer.duration());
#endif
       
    }
};

class PathCollector
{
    int worker_num;
    // node_buffer
    std::vector<MessageBuffer*> node_local_fp;
    std::mutex node_local_fp_lock;
    // 线程buffer
    MessageBuffer** thread_local_fp;
public:
    PathCollector(int worker_num_param)
    {
        this->worker_num = worker_num_param;
        thread_local_fp = new MessageBuffer*[worker_num];
        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            thread_local_fp[w_i] = nullptr;
        }
    }
    ~ PathCollector()
    {
        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            if (thread_local_fp[w_i] != nullptr)
            {
                delete thread_local_fp[w_i];
            }
        }
        delete []thread_local_fp;
        for (auto buf : node_local_fp)
        {
            if (buf != nullptr)
            {
                delete buf;
            }
        }
    }

    void add_footprint(Footprint ft, int worker)
    {
        // 线程buffer
        MessageBuffer* &fp_buffer = thread_local_fp[worker];
        if (fp_buffer == nullptr)
        {
            fp_buffer = new MessageBuffer(sizeof(Footprint) * FOOT_PRINT_CHUNK_SIZE);
        }
        fp_buffer->write(&ft);
        // 游走数据达到65535
        if (fp_buffer->count == FOOT_PRINT_CHUNK_SIZE)
        {
            node_local_fp_lock.lock();
            node_local_fp.push_back(fp_buffer);
            fp_buffer = nullptr;
            node_local_fp_lock.unlock();
        }
    }

    PathSet* assemble_path(walker_id_t walker_begin)
    {
        Timer timer;
        //flush thread local fp
        // 将线程buffer里保存的foot_print数据写入到node_buffer_foot_print,不同的是，这里的线程buffer里的数据可能是不满的
        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            if (this->thread_local_fp[w_i] != nullptr)
            {
                this->node_local_fp.push_back(thread_local_fp[w_i]);
                this->thread_local_fp[w_i] = nullptr;
            }
        }
        // 那其实统计的时候可以放到这里

        /****************************************************这一段的作用是统计本台机器该发送到对应机器的数量*****************************************************/
        //shuffle the footprints
        partition_id_t partition_num = get_mpi_size();
        partition_id_t partition_id = get_mpi_rank();
        // 这点的walker初始化划分是不一样的，这里的分配是按照walker_id % partition的方式，这样的方式是为了负载更加均衡
        auto get_walker_partition_id = [&] (walker_id_t walker)
        {
            return walker % partition_num;
        };
        // 发送到各个节点的的 foot_print 总数
        size_t send_fp_num[partition_num];
        std::fill(send_fp_num, send_fp_num + partition_num, 0);
        // 这个progress是共享变量
        size_t progress = 0;
#pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            size_t next_workload;
            // 统计本线程负责统计的数据，要发送给各个节点的消息数
            size_t local_counter[partition_num];
            std::fill(local_counter, local_counter + partition_num, 0);
            // 开始遍历node_local_fp(是个数组)里的所有foot_print
            // next_workload是线程自己的私有变量，故这个next_workload会跳变
            while ((next_workload =  __sync_fetch_and_add(&progress, 1)) < node_local_fp.size())
            {
                MessageBuffer* buf = node_local_fp[next_workload];
                Footprint* begin = (Footprint*)buf->data; 
                Footprint* end = begin + buf->count;
                // 一个一个地遍历
                for (Footprint *fp = begin; fp < end; fp++)
                {
                    // 根据walker.id统计
                    local_counter[get_walker_partition_id(fp->walker)]++;
                }
            }
            // 所以才会把local_counter[p_i]进行加和，得到的是本节点该向该节点发送的数据
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                __sync_fetch_and_add(&send_fp_num[p_i], local_counter[p_i]);
            }
        }

        // 单线程执行
        // 这台机器的总foot_print 发送量
        size_t tot_send_fp_num = 0;
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            tot_send_fp_num += send_fp_num[p_i];
        }
        /***************************************send_buffer是共享变量****************************************/
        // 待发送foot_print
        Footprint* send_buf = new Footprint[tot_send_fp_num];
        size_t partition_progress[partition_num];
        // 发送到节点0的foot_print在send_buf里面的起始位置
        partition_progress[0] = 0;
        for (partition_id_t p_i = 1; p_i < partition_num; p_i++)
        {
            partition_progress[p_i] = partition_progress[p_i - 1] + send_fp_num[p_i - 1];
        }
        // 发送到节点0的foot_print在send_buf里面的起始位置
        size_t send_fp_pos[partition_num];
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            send_fp_pos[p_i] = partition_progress[p_i];
        }
        /**************************************************************************************************************************************************/
        /************************************************这一段的作用把数据放到对应的send_buffer里面并发送和接收***************************************************/
        progress = 0;
#pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            size_t next_workload;
            const size_t LOCAL_BUF_SIZE = THREAD_LOCAL_BUF_CAPACITY;
            // 本线程待发送到节点p_i的buffer
            MessageBuffer* local_buf[partition_num];
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                local_buf[p_i] = new MessageBuffer(LOCAL_BUF_SIZE * sizeof(Footprint));
            }
            // 给定一个机器id,把本条线程负责的local_buffer里面的数据搬运到send_buffer
            auto flush_local_buf = [&] (partition_id_t p_i)
            {
                if (local_buf[p_i]->count != 0)
                {
                    // 因为其他的线程也在做这件事，所以要使用原子加
                    size_t start_pos = __sync_fetch_and_add(&partition_progress[p_i], (size_t)local_buf[p_i]->count);
                    // 把local_buffer里面的数据复制到send_buffer
                    memcpy(send_buf + start_pos, local_buf[p_i]->data, local_buf[p_i]->count * sizeof(Footprint));
                    local_buf[p_i]->count = 0;
                }
            };
            while ((next_workload =  __sync_fetch_and_add(&progress, 1)) < node_local_fp.size())
            {
                // 还是在遍历node_local_fp
                MessageBuffer* &buf = node_local_fp[next_workload];
                Footprint* begin = (Footprint*)buf->data; 
                Footprint* end = begin + buf->count;
                for (Footprint* fp = begin; fp < end; fp++)
                {
                    partition_id_t p_i = get_walker_partition_id(fp->walker);
                    // 真实写入到local_buf
                    local_buf[p_i]->write(fp);
#ifdef UNIT_TEST
                    local_buf[p_i]->self_check<Footprint>();
#endif
                    // 如果待发送目标节点的local_buffer满了，就把他刷新到send_buffer里面
                    if (local_buf[p_i]->count == LOCAL_BUF_SIZE)
                    {
                        flush_local_buf(p_i);
                    }
                }
                delete buf;
                buf = nullptr;
            }
            // 处理到最后，如果还有些local_buffer没满，把那些没满的也刷新到send_buffer里面
            for (partition_id_t p_i = 0; p_i < partition_num; p_i++)
            {
                flush_local_buf(p_i);
                delete local_buf[p_i];
            }
        }
        // 把node_local_fp清空
        node_local_fp.clear();
        // 向每个节点发送的数据总量
        size_t glb_send_fp_num[partition_num];
        MPI_Allreduce(send_fp_num, glb_send_fp_num, partition_num, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        // 本台机器总的接收总量，向他发送了多少个，就要接收多少个
        size_t total_recv_fp_num = glb_send_fp_num[partition_id];
		size_t recv_fp_pos = 0;
        Footprint* recv_buf = new Footprint[total_recv_fp_num];
        std::thread send_thread([&](){
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t dst = (partition_id + step) % partition_num;
                // 总的发送字节数
                size_t tot_send_sz = send_fp_num[dst] * sizeof(Footprint);
#ifdef UNIT_TEST
                const int max_single_send_sz = (1 << 12) / sizeof(Footprint) * sizeof(Footprint);
#else
                // 单次发送数据总量
                const int max_single_send_sz = (1 << 28) / sizeof(Footprint) * sizeof(Footprint);
#endif
                // 发送数据的起始位置
                void* send_data = send_buf + send_fp_pos[dst];
                while (true)
                {
                    // 首先告诉对方发送数据总量
                    MPI_Send(&tot_send_sz, 1, get_mpi_data_type<size_t>(), dst, 0, MPI_COMM_WORLD);
                    if (tot_send_sz == 0)
                    {
                        break;
                    }
                    size_t send_sz = std::min((size_t)max_single_send_sz, tot_send_sz);
                    tot_send_sz -= send_sz;
                    MPI_Send(send_data, send_sz, get_mpi_data_type<char>(), dst, 0, MPI_COMM_WORLD);
                    send_data = (char*)send_data + send_sz;
                }
            }
        });
        std::thread recv_thread([&](){
            for (partition_id_t step = 0; step < partition_num; step++)
            {
                partition_id_t src = (partition_id + partition_num - step) % partition_num;
                while (true)
                {
                    // 首先收到该接收多少数据
                    size_t remained_sz;
                    MPI_Recv(&remained_sz, 1, get_mpi_data_type<size_t>(), src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (remained_sz == 0)
                    {
                        break;
                    }
                    MPI_Status recv_status;
                    MPI_Probe(src, 0, MPI_COMM_WORLD, &recv_status);
                    int sz;
                    MPI_Get_count(&recv_status, get_mpi_data_type<char>(), &sz);

                    MPI_Recv(recv_buf + recv_fp_pos, sz, get_mpi_data_type<char>(), src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    recv_fp_pos += sz / sizeof(Footprint);
                }
            }
        });
        send_thread.join();
        recv_thread.join();
        delete []send_buf;

        /**************************************************************************************************************************************************/
        /***************************************************分配任务，实际上是把walkerid相同的foot_print放一起**************************************************/
        //shuffle task locally
        // PARALLEL_CHUNK_SIZE 定义了多线程调度的粒度
        const size_t large_step_size = PARALLEL_CHUNK_SIZE * worker_num;
        // 任务数量, 计算出来的task_num 正好介于total_recv_fp_num/large_step_size 或者 total_recv_fp_num/large_step_size + 1
        const size_t task_num = (total_recv_fp_num + large_step_size - 1) / large_step_size;
        // 线程i负责的任务的起始位置和结束位置(foot_print的位置)，其实是一张二维表，但是里面应该有很多的空间没用
        Footprint** task_begin[worker_num];
        Footprint** task_end[worker_num];
        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            // 线程对应
            task_begin[w_i] = new Footprint*[task_num];
            task_end[w_i] = new Footprint*[task_num];
        }
        // 每台机器上的每一个线程负责的walker.我猜应该是用多个线程把它汇集寄来,还回值一定是介于0-22
        auto get_task_partition = [&] (walker_id_t walker)
        {
            // 如果没有除法，0，23，23*2 会分在一个线程，有了除法0.1.2.3，4*23.4*23+1会分在一个线程
            return (walker - walker_begin) / partition_num % worker_num;
        };
        progress = 0;
#pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            size_t next_workload;
            size_t task_counter[worker_num];
            // 把线程worker_id负责的任务给领出来
            Footprint* task_pos[worker_num];
            Footprint* local_fp = new Footprint[large_step_size];
            // 多个线程共同的操作
            while ((next_workload =  __sync_fetch_and_add(&progress, 1)) < task_num)
            {
                // 线程之间获取数据,注意这个begin的跨度
                Footprint* begin = recv_buf + next_workload * large_step_size; 
                Footprint* end = recv_buf + std::min((next_workload + 1) * large_step_size, total_recv_fp_num);
                // 处理的foot_print数量
                size_t work_load_size = end - begin;
                // 开始和结束位置的foot_print
                Footprint* local_begin = local_fp;
                Footprint* local_end = local_fp + work_load_size;
                // 把数据从recv_buffer里面copy到local_fp，然后在排好序写会recv_buffer
                memcpy(local_begin, begin, work_load_size * sizeof(Footprint));
                // 每个线程该负责的foot_print
                std::fill(task_counter, task_counter + worker_num, 0);
                for (Footprint *fp = local_begin; fp < local_end; fp++)
                {
                    // 对应线程要完成的任务数(以foot_print计数),把这一小段的foot_print分配给不同的线程
                    task_counter[get_task_partition(fp->walker)]++;
                }
                // 当前任务的位置
                Footprint* current_task_pos = begin;
                // 记录线程该顾泽的foot_print的开始位置和结束位置
                for (partition_id_t p_i = 0; p_i < worker_num; p_i++)
                {
                    // 线程p_i的任务next_workload
                    task_begin[p_i][next_workload] = current_task_pos;
                    task_pos[p_i] = current_task_pos;
                    // 跳到下一个线程该执行的位置
                    current_task_pos += task_counter[p_i];
                    task_end[p_i][next_workload] = current_task_pos;
                }
                for (Footprint *fp = local_begin; fp < local_end; fp++)
                {
                    partition_id_t task_p = get_task_partition(fp->walker);
                    // 将foot_print放置于对应的任务区
                    *task_pos[task_p] = *fp;
                    task_pos[task_p]++;
                }
            }
            delete []local_fp;
        }

        //do tasks
        vertex_id_t* path_set[worker_num];
        vertex_id_t** path_begin[worker_num];
        vertex_id_t** path_end[worker_num];
        walker_id_t* walker_id[worker_num];
        step_t* path_length[worker_num];
        walker_id_t path_num[worker_num];

        // 得到walker的内部编号最大值
        auto get_walker_local_idx = [&] (walker_id_t walker)
        {
            // 0 - 4*23 - 1 都为0, 4 * 23 - 8 * 23 -1为1，相当于以4*23个大小打包
            return (walker - walker_begin) / partition_num / worker_num;
        };
#pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            // 这个线程处理的最大的walker_id
            walker_id_t max_walker_id = 0;
            bool has_any_walker = false;
            // 处理了多少个foot_print
            size_t step_num = 0;
            // 任务可以理解为路径，也就是一个walker
            for (size_t t_i = 0; t_i < task_num; t_i++)
            {
                Footprint* begin = task_begin[worker_id][t_i];
                Footprint* end = task_end[worker_id][t_i];
                for (Footprint* fp = begin; fp < end; fp++)
                {
                    has_any_walker = true;
                    // 求处理区间的最大walker_id
                    max_walker_id = std::max(max_walker_id, fp->walker);
                    // 跳变里面有多少个foot_print
                    step_num++;
                }
            }
            // 处理的最大walker的本机内部编号
            walker_id_t max_walker_idx = get_walker_local_idx(max_walker_id);
            // std::cout << "max_walker_idx = " << max_walker_idx << std::endl;
            // 记录这个线程处理的walker没有
            // max_walker_idx只是索引，线程负责的walker数量thread_walker_num 实际比它大1
            walker_id_t thread_walker_num = has_any_walker ? max_walker_idx + 1 : 0;

            // path_set 相当于一个计数器
            path_set[worker_id] = new vertex_id_t[step_num];
            // 线程worker_id负责的path数量为 thread_walker_num, path_begin和path_end都保存的是顶点的地址
            path_begin[worker_id] = new vertex_id_t*[thread_walker_num];
            path_end[worker_id] = new vertex_id_t*[thread_walker_num];
            walker_id[worker_id] = new walker_id_t[thread_walker_num];
            path_length[worker_id] = new step_t[thread_walker_num];
            // thread_walker_num 线程负责的walker数量
            path_num[worker_id] = thread_walker_num;

            for (walker_id_t w_i = 0; w_i < thread_walker_num; w_i++)
            {
                path_length[worker_id][w_i] = 0;
                // 第i段第j条路径的walker id , 还原对应的真实id
                walker_id[worker_id][w_i] = w_i * partition_num * worker_num + partition_num * worker_id + partition_id + walker_begin;
            }
            for (size_t t_i = 0; t_i < task_num; t_i++)
            {
                Footprint* begin = task_begin[worker_id][t_i];
                Footprint* end = task_end[worker_id][t_i];
                for (Footprint* fp = begin; fp < end; fp++)
                {
                    walker_id_t idx = get_walker_local_idx(fp->walker);
                    path_length[worker_id][idx]++;
                }
            }
            size_t step_counter = 0;
            for (walker_id_t w_i = 0; w_i < thread_walker_num; w_i++)
            {
                // 开辟好空间直接放数据
                path_begin[worker_id][w_i] = path_set[worker_id] + step_counter;
                step_counter += path_length[worker_id][w_i];
                path_end[worker_id][w_i] = path_set[worker_id] + step_counter;
            }
            // 每个线程该负责的任务
            for (size_t t_i = 0; t_i < task_num; t_i++)
            {
                Footprint* begin = task_begin[worker_id][t_i];
                Footprint* end = task_end[worker_id][t_i];
                for (Footprint* fp = begin; fp < end; fp++)
                {
                    walker_id_t idx = get_walker_local_idx(fp->walker);
                    // 线程内部负责的walker要进行重新编号
                    *(path_begin[worker_id][idx] + fp->step) = fp->vertex;
                }
            }
        }
        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            delete []task_begin[w_i];
            delete []task_end[w_i];
        }
        delete []recv_buf;

        /****************************************************还回结果*****************************************/
        PathSet* ps = new PathSet();
        ps->seg_num = worker_num;
        // ps->path_set = new vertex_id_t*[worker_num];
        ps->walker_id = new walker_id_t*[worker_num];
        ps->path_begin = new vertex_id_t**[worker_num];
        ps->path_end = new vertex_id_t**[worker_num];
        ps->path_length = new step_t*[worker_num];
        ps->path_num = new walker_id_t[worker_num];
        for (int w_i = 0; w_i < worker_num; w_i++)
        {
            // ps->path_set[w_i] = path_set[w_i];
            ps->walker_id[w_i] = walker_id[w_i];
            ps->path_begin[w_i] = path_begin[w_i];
            ps->path_end[w_i] = path_end[w_i];
            ps->path_length[w_i] = path_length[w_i];
            ps->path_num[w_i] = path_num[w_i];
        }
#ifndef UNIT_TEST
        printf("[p%d] finish assembling in %lf seconds\n",get_mpi_rank(), timer.duration());
#endif
        return ps;
    }
};
