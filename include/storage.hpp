
#pragma once

#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include <vector>
#include <unordered_map>

#include "type.hpp"
using namespace std;

template<typename T>
void read_graph(const char* fname, int partition_id, int partition_num, T* &edge, size_t &e_num)
{
    FILE *f = fopen(fname, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    size_t total_size = ftell(f);
    // sizeof T is 12
    size_t total_e_num = total_size / sizeof(T);
    // std::cout << "size of T = " << sizeof(T);
    e_num = total_e_num / partition_num;

    if (partition_id == partition_num -1)
    {
        e_num += total_e_num % partition_num;
    }
    size_t f_offset = sizeof(T) * (total_e_num / partition_num) * partition_id; 

    edge = new T[e_num];
    fseek(f, f_offset, SEEK_SET);
    auto ret = fread(edge, sizeof(T), e_num, f);
    assert(ret == e_num);
    fclose(f);
}

template<typename T>
void g_read_graph(const char* fname, T* &edge, size_t &e_num)
{
    FILE *f = fopen(fname, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);

    size_t total_size = ftell(f);
    // sizeof T is 12
    size_t total_e_num = total_size / sizeof(T);
    // std::cout << "size of T = " << sizeof(T);

    e_num = total_e_num ;

    size_t f_offset = 0;

    edge = new T[e_num];
    fseek(f, f_offset, SEEK_SET);
    auto ret = fread(edge, sizeof(T), e_num, f);
    assert(ret == e_num);
    fclose(f);
}

template<typename T>
void read_graph_common_neighbour(const char* fname, T* &edge, unordered_map<com_neighbour_t, int> &mp, partition_id_t *vertex_partition_id, partition_id_t &local_partition_id)
{
    FILE *f = fopen(fname, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);

    size_t total_size = ftell(f);
    // sizeof T is 12
    size_t total_e_num = total_size / sizeof(T);
    // std::cout << "total_e_num = " << total_e_num << std::endl;
    edge = new T[total_e_num];
    fseek(f, 0, SEEK_SET);
    auto ret = fread(edge, sizeof(T), total_e_num, f);
    // std::cout << "ret = " << ret << std::endl;
    assert(ret == total_e_num);
    fclose(f);
    U u1, u2;
    for(edge_id_t e_i = 0; e_i < total_e_num; e_i++)
    {
        if(vertex_partition_id[edge[e_i].src] == local_partition_id)
        {
            u1 = {edge[e_i].src, edge[e_i].dst};
            mp[u1.key] = edge[e_i].data;
        }
        if(vertex_partition_id[edge[e_i].dst] == local_partition_id)
        {
            u2 = {edge[e_i].dst, edge[e_i].src};
            mp[u2.key] = edge[e_i].data;
        }
        // U u1 = {edge[e_i].src, edge[e_i].dst};
        // U u2 = { edge[e_i].dst, edge[e_i].src};
        // mp[u1.key] = edge[e_i].data;
        // mp[u2.key] = edge[e_i].data;

        // com_neighbour_t src_dst_com_nei = edge[e_i].src;
        // src_dst_com_nei = (src_dst_com_nei << 32 ) | edge[e_i].dst;
        // com_neighbour_t dst_src_com_nei = edge[e_i].dst;
        // dst_src_com_nei = (dst_src_com_nei << 32) | edge[e_i].src;
        // mp[src_dst_com_nei] = edge[e_i].data;
        // mp[dst_src_com_nei] = edge[e_i].data;
    }
}

template<typename T>
void write_graph(const char* fname, const T* es, const size_t e_num)
{
    FILE *out_f = fopen(fname, "w");
    assert(out_f != NULL);
    auto ret = fwrite(es, sizeof(T), e_num, out_f);
    assert(ret == e_num);
    fclose(out_f);
}

size_t next_endline_pos(FILE *f)
{
    size_t current_pos = ftell(f);
    while (true)
    {
        char ch;
        auto ret = fread(&ch, 1, 1, f);
        if (ret != 1 || ch == '\n')
        {
            break;
        }
        current_pos++;
    }
    return current_pos;
}

std::vector<size_t> partition_text_file(const char* fname, int partition_num)
{
    std::vector<size_t> partition_end;
    FILE *f = fopen(fname, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    size_t total_size = ftell(f);
    for (int p_i = 0; p_i < partition_num; p_i++)
    {
        size_t f_offset = total_size / partition_num * (p_i + 1);
        if (f_offset >= total_size)
        {
            partition_end.push_back(f_offset);
        } else
        {
            fseek(f, f_offset, SEEK_SET);
            partition_end.push_back(next_endline_pos(f));
        }
    }
    fclose(f);
    return partition_end;
}

bool read_edge_txt(FILE* f, Edge<EmptyData>* edge)
{
    return (2 == fscanf(f, "%u %u", &edge->src, &edge->dst));
}

bool read_edge_txt(FILE* f, Edge<real_t>* edge)
{
    return (3 == fscanf(f, "%u %u %f", &edge->src, &edge->dst, &edge->data));
}

bool read_edge_txt(FILE* f, Edge<uint32_t>* edge)
{
	return (3 == fscanf(f, "%u %u %u", &edge->src, &edge->dst, &edge->data));
}

template<typename T>
bool read_edge_txt(FILE* f, Edge<T>* edge)
{
    fprintf(stderr, "Edge type doesn't support reading from text\n");
    exit(1);
}

template<typename T>
void read_edgelist(const char* fname, int partition_id, int partition_num, Edge<T>* &edge, size_t &e_num)
{
    std::vector<size_t> partition_end = partition_text_file(fname, partition_num);
    size_t begin = (partition_id == 0 ? 0 : partition_end[partition_id - 1]);
    size_t end = partition_end[partition_id];
    FILE *f = fopen(fname, "r");
    assert(f != NULL);

    Edge<T> temp;
    e_num = 0;
    fseek(f, begin, SEEK_SET);
    while (ftell(f) < end)
    {
        if (read_edge_txt(f, &temp))
        {
            e_num++;
        }
    }
    edge = new Edge<T>[e_num];

    size_t e_i = 0;
    fseek(f, begin, SEEK_SET);
    while (ftell(f) < end)
    {
        if (read_edge_txt(f, &temp))
        {
            edge[e_i] = temp;
            e_i++;
        }
    }

    fclose(f);
}

void print_edge(FILE* f, const Edge<EmptyData>* edge)
{
   fprintf(f, "%u %u\n", edge->src, edge->dst);
}

void print_edge(FILE* f, const Edge<real_t>* edge)
{
   fprintf(f, "%u %u %f\n", edge->src, edge->dst, edge->data);
}

template<typename T>
void print_edge(FILE* f, const Edge<T>* edge)
{
    fprintf(stderr, "Edge type doesn't support writing to text\n");
    exit(1);
}

template<typename T>
void write_edgelist(const char* fname, const Edge<T>* es, const size_t e_num)
{
    FILE *out_f = fopen(fname, "w");
    assert(out_f != NULL);
    for (size_t e_i = 0; e_i < e_num; e_i++)
    {
        print_edge(out_f, es + e_i);
    }
    fclose(out_f);
}
