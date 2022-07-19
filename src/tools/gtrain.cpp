#include <assert.h>

#include <iostream>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <ctime>

#include "storage.hpp"
#include "type.hpp"
#include "option_helper.hpp"
#include "graph.hpp"


int thread_num=1;


class GtrainOptionHelper : public OptionHelper
{
private:
    args::ValueFlag<std::string> input_flag;
    args::ValueFlag<std::string> lpfile_flag;
    args::ValueFlag<std::string> test_edges_flag;
    args::ValueFlag<std::string> test_labels_flag;
    args::ValueFlag<float> sample_flag;
    args::ValueFlag<vertex_id_t> v_num_flag;
    args::ValueFlag<int> thread_flag;
public:
    std::string input;
    std::string lpfile;
    std::string test_edges;
    std::string test_labels;
    float sample;
    vertex_id_t v_num;
    int thread;
    GtrainOptionHelper() :
        input_flag(parser, "input", "input graph path", {'i'}),
        lpfile_flag(parser, "lpfile", "train graph file path", {'o'}),
        test_edges_flag(parser, "test_edges", "test edegs path", {'e'}),
        test_labels_flag(parser, "test_labels", "test labels path", {'l'}),
        sample_flag(parser, "sample", "Inout hyperparameter.", {'s'}),
        v_num_flag(parser, "vertex", "vertex number", {'v'}),
        thread_flag(parser, "thread", "thread number for csr", {'t'})
    {
    }

    virtual void parse(int argc, char **argv)
    {
        OptionHelper::parse(argc, argv);

        assert(input_flag);
        input = args::get(input_flag);

        assert(lpfile_flag);
        lpfile = args::get(lpfile_flag);

        assert(test_edges_flag);
        test_edges = args::get(test_edges_flag);

        assert(test_labels_flag);
        test_labels = args::get(test_labels_flag);

        assert(sample_flag);
        sample = args::get(sample_flag);

        assert(v_num_flag);
        v_num = args::get(v_num_flag);

        assert(thread_flag);
        thread = args::get(thread_flag);

    }
};

template<typename edge_data_t>
void build_edge_container(vertex_id_t v_num,Edge<edge_data_t> *edges, edge_id_t local_edge_num, EdgeContainer<edge_data_t> *ec, vector<vertex_id_t>& vertex_out_degree)
{

    ec->adj_lists = new AdjList<edge_data_t>[v_num];

    ec->adj_units = new AdjUnit<edge_data_t>[local_edge_num];
    edge_id_t chunk_edge_idx = 0;
    for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
    {
        ec->adj_lists[v_i].begin = ec->adj_units + chunk_edge_idx;
        ec->adj_lists[v_i].end = ec->adj_lists[v_i].begin;

        chunk_edge_idx += vertex_out_degree[v_i];
    }
    printf("csr adjL ok\n");
    for (edge_id_t e_i = 0; e_i < local_edge_num; e_i++)
    {
        auto e = edges[e_i];
        auto ep = ec->adj_lists[e.src].end ++;

        ep->neighbour = e.dst;

    }
}

void read_txt_edges(const char* input_path, std::vector<Edge<EmptyData> > &edges)
{
    FILE *f = fopen(input_path, "r");
    assert(f != NULL);
    vertex_id_t src, dst;
    while (2 == fscanf(f, "%u\t%u", &src, &dst))
    {
        edges.push_back(Edge<EmptyData>(src, dst));
    }
    fclose(f);
}

void read_txt_edges(const char* input_path, std::vector<Edge<real_t> > &edges)
{
    FILE *f = fopen(input_path, "r");
    vertex_id_t src, dst;
    real_t weight;
    while (3 == fscanf(f, "%u %u %f", &src, &dst, &weight))
    {
        edges.push_back(Edge<real_t>(src, dst, weight));
    }
    fclose(f);
}

void read_txt_edges(const char* input_path, std::vector<Edge<int> > &edges)
{
    FILE *f = fopen(input_path, "r");
    vertex_id_t src, dst;
    int weight;
    while (3 == fscanf(f, "%u\t%u\t%d", &src, &dst, &weight))
    {
        edges.push_back(Edge<int>(src, dst, weight));
    }
    fclose(f);
}

void read_txt_edges(const char* input_path, std::vector<Edge<vertex_id_t> > &edges)
{
    FILE *f = fopen(input_path, "r");
    vertex_id_t src, dst;
    vertex_id_t weight;
    while (3 == fscanf(f, "%u\t%u\t%u", &src, &dst, &weight))
    {
        edges.push_back(Edge<vertex_id_t>(src, dst, weight));
    }
    fclose(f);
}


template<typename edge_data_t>
void sort_csr(EdgeContainer<edge_data_t>* csr, vertex_id_t v_num,edge_id_t local_edge_num, std::vector<vertex_id_t>&vertex_out_degree) {
#pragma omp parallel for  num_threads(thread_num)
    for (vertex_id_t v_i = 0; v_i < v_num; v_i++) {
        std::sort(csr->adj_lists[v_i].begin, csr->adj_lists[v_i].end,
            [](const AdjUnit<edge_data_t> a, const AdjUnit<edge_data_t> b) {return a.neighbour < b.neighbour; });
    }

//     AdjUnit<edge_data_t>* new_adju = new AdjUnit<edge_data_t>[local_edge_num];

// #pragma omp parallel for  num_threads(thread_num)
//     for (vertex_id_t v_i = 0; v_i < v_num; v_i++) {

//         AdjUnit<edge_data_t>* t_begin = new_adju + (csr->adj_lists[v_i].begin - csr->adj_units);
//         AdjUnit<edge_data_t>* t_end = t_begin;

//         if(csr->adj_lists[v_i].end - csr->adj_lists[v_i].begin == 0){
//             csr->adj_lists[v_i].begin = t_begin;
//             csr->adj_lists[v_i].end = t_end;
//             continue;
//         }

//         t_begin->neighbour = csr->adj_lists[v_i].begin->neighbour;
//         t_begin->data = csr->adj_lists[v_i].begin->data;
//         t_end ++;
//         for(auto e= csr->adj_lists[v_i].begin+1; e< csr->adj_lists[v_i].end; e++){
//             if(e->neighbour != (e-1)->neighbour){
//                 t_end->neighbour = e->neighbour;
//                 t_end->data = e->data;
//                 t_end++;
//             }
//         }
//         csr->adj_lists[v_i].begin = t_begin;
//         csr->adj_lists[v_i].end = t_end;
//         vertex_out_degree[v_i] = t_end - t_begin;
//     }

//     delete csr->adj_units;
//     csr->adj_units = new_adju;
}

template<typename T>
AdjUnit<T>* get_unit_csr(vertex_id_t src, vertex_id_t dst, EdgeContainer<T>* csr) {
    AdjList<T>* adj_list = &csr->adj_lists[src];
    AdjUnit<T> target;
    target.neighbour = dst;
    auto edge = std::lower_bound(adj_list->begin, adj_list->end, target, [](const AdjUnit<T> & a, const AdjUnit<T> & b) { return a.neighbour < b.neighbour; });

    if (edge != adj_list->end && dst == edge->neighbour)
        return edge;
    else
        return nullptr;
}

bool try_add_neg_edge(vertex_id_t src,vertex_id_t dst,std::vector<std::vector<vertex_id_t>>& neg_edge_csr)
{
    auto it = std::find(neg_edge_csr[src].begin(),neg_edge_csr[src].end(),dst);
    if(it == neg_edge_csr[src].end()){
        neg_edge_csr[src].push_back(dst);
        return true;
    }else
        return false;
}

edge_id_t get_big_rand(edge_id_t range){
    if(range < RAND_MAX)return rand() % range; 
    else{
        edge_id_t round = range / RAND_MAX; 
        edge_id_t mod = range % RAND_MAX; 
        edge_id_t t = rand() % round + 1;
        if(t== round)return t* RAND_MAX + rand()%mod;
        else return t* RAND_MAX + rand() % RAND_MAX;
    }
}

template<typename edge_data_t>
void gtrain(const char* input_path, const char* lpfile, const char* test_edges, const char* test_labels,float sample,vertex_id_t v_num)
{
    std::vector<Edge<edge_data_t> > edges;
    read_txt_edges(input_path, edges);

    std::vector<Edge<edge_data_t> >* edge_ptr = &edges;
    std::vector<Edge<edge_data_t>> undirected_edges;
    edge_id_t edge_num = edges.size();

    printf("input file has %zu edges\n",edge_num);
     if (true)
    {
        undirected_edges.resize(edges.size() * 2);
#pragma omp parallel for num_threads(thread_num)
        for (edge_id_t e_i = 0; e_i < edges.size(); e_i++)
        {

            undirected_edges[e_i * 2] = edges[e_i];

            undirected_edges[e_i * 2 + 1] = edges[e_i];
            std::swap(undirected_edges[e_i * 2 + 1].src, undirected_edges[e_i * 2 + 1].dst);
        }
        edge_ptr = &undirected_edges;
        printf("read as undirected ok\n");
    }
    else
        printf("read ok\n");

    // write_graph(output_path, edges.data(), edges.size());
    std::vector<vertex_id_t>vertex_out_degree(v_num);
    printf("vertex degree size: %lu \n",vertex_out_degree.size());
   
    for (edge_id_t e_i = 0; e_i < edge_ptr->size(); e_i++) 
    {

        vertex_out_degree[(*edge_ptr)[e_i].src]++;
    }
    printf("cal degree ok\n");

    vertex_id_t count = 0;
    vertex_id_t count_1 = 0;
    for(vertex_id_t v_i = 0 ; v_i < vertex_out_degree.size();v_i++){
        if(vertex_out_degree[v_i]==0){
            // printf("%d degree = 0\n",v_i);
            count++;
        }
        if(vertex_out_degree[v_i]==1){
            count_1 ++;
        }
    }
    printf("origin graph degree 0 count %u degree 1 count %u\n",count,count_1);

    EdgeContainer<edge_data_t>*graph = new EdgeContainer<edge_data_t>();
    build_edge_container<edge_data_t>(v_num,(*edge_ptr).data(), (*edge_ptr).size(), graph, vertex_out_degree);
    printf("build csr ok\n");
    sort_csr(graph,v_num,(*edge_ptr).size(),vertex_out_degree);
    printf("sort csr ok\n");

    // for(int i=0;i<v_num;i++){
    //     printf("%u\n",i);
    //     for(auto a=graph->adj_lists[i].begin;a<graph->adj_lists[i].end;a++){
    //         printf("%u|%d ",a->neighbour,a->data);
    //     }
    //     printf("\n");
    // }


    edge_id_t npos = sample * edges.size();
    edge_id_t nneg = sample * edges.size();

    std::vector<Edge<edge_data_t>>pos_edges(npos);
    edge_id_t pos_tail = edge_num -1;
    edge_id_t pos_p = 0;
    std::vector<Edge<edge_data_t>>neg_edges(nneg);
    edge_id_t neg_p = 0;

    srand((int)time(0));

    

// #pragma omp parallel for num_threads(thread_num)
    for(edge_id_t e_i = 0; e_i < npos; e_i ++){

        edge_id_t ep = get_big_rand(pos_tail+1);

        while(vertex_out_degree[edges[ep].src]<=1 || vertex_out_degree[edges[ep].dst]<=1){
            ep = get_big_rand(pos_tail+1);
        }

        pos_edges[pos_p].src = edges[ep].src;
        pos_edges[pos_p].dst = edges[ep].dst;

        vertex_out_degree[edges[ep].src]--;
        vertex_out_degree[edges[ep].dst]--;

        std::swap(edges[ep].src,edges[pos_tail].src);
        std::swap(edges[ep].dst,edges[pos_tail].dst);
        std::swap(edges[ep].data,edges[pos_tail].data);

        pos_p ++;
        pos_tail --;
    }

    std::vector<std::vector<vertex_id_t>> neg_edge_csr(v_num);
    for(edge_id_t e = 0; e< nneg; e++){

        vertex_id_t neg_src = get_big_rand(v_num);
        vertex_id_t neg_dst = get_big_rand(v_num);
 
        while(get_unit_csr(neg_src,neg_dst,graph)!=nullptr || try_add_neg_edge(neg_src,neg_dst,neg_edge_csr)== false)
        {
            neg_src = get_big_rand(v_num);
            neg_dst = get_big_rand(v_num);
        }
        neg_edges[neg_p].src = neg_src;
        neg_edges[neg_p].dst = neg_dst;
        neg_p ++ ;
    }

    FILE* flpfile = fopen(lpfile,"w");
    
    vector<vertex_id_t> train_graph_degree(v_num,0);
    edge_id_t train_edge = 0;
    for(edge_id_t e=0; e<=pos_tail;e++)
    {
        fprintf(flpfile,"%u %u %u\n",edges[e].src,edges[e].dst,edges[e].data);
        train_edge ++;
        train_graph_degree[edges[e].src]++;
        train_graph_degree[edges[e].dst]++;
    }

    count = 0;
    for(vertex_id_t v_i = 0 ; v_i < train_graph_degree.size();v_i++){
        if(train_graph_degree[v_i]==0){
            // printf("%d degree = 0\n",v_i);
            count++;
        }
    }
    printf("train degree  0 count %u\n",count);
    
    fclose(flpfile);
    printf("%zu train edges\n", pos_tail+1);

    FILE * ftest_edges = fopen(test_edges,"w");
    FILE * ftest_labels = fopen(test_labels,"w");

    for(edge_id_t e = 0; e<npos;e++ ){
        fprintf(ftest_edges,"%u %u\n",pos_edges[e].src,pos_edges[e].dst);
        fprintf(ftest_labels,"1\n");
    }
    printf("npos edge : %lu\n",npos);
    for(edge_id_t e =0; e<nneg; e++){
        fprintf(ftest_edges,"%u %u\n",neg_edges[e].src,neg_edges[e].dst);
        fprintf(ftest_labels,"0\n");

    }
    printf("nneg edge: %lu\n",nneg);
    fclose(ftest_edges);
    fclose(ftest_labels);

}

int main(int argc, char** argv)
{
    Timer timer;
    srand(time(NULL));
    GtrainOptionHelper opt;
    opt.parse(argc, argv);
    thread_num = opt.thread;
    if(thread_num==0)thread_num=1;
    printf("thread_num for csr: %d\n",thread_num);
    if(opt.sample>1){
        printf("[error] sample > 1.0\n");
        exit(1);
    }
    // if (opt.static_comp.compare("weighted") == 0)
    // {
    //     // gcn<real_t>(opt.input_path.c_str(), opt.output_path.c_str(),opt.make_undirected,opt.v_num);
    //     gcn<int>(opt.input_path.c_str(), opt.output_path.c_str(),opt.make_undirected,opt.v_num);
    // } else
    // {
    gtrain<vertex_id_t>(opt.input.c_str(), opt.lpfile.c_str(),opt.test_edges.c_str(),opt.test_labels.c_str(),opt.sample,opt.v_num);
    // }

    printf("[gtrain] time: %f \n",timer.duration());
	return 0;
}
