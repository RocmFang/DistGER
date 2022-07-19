#include <assert.h>

#include <iostream>
#include <utility>
#include <type_traits>
#include <algorithm>

#include "storage.hpp"
#include "type.hpp"
#include "option_helper.hpp"
#include "graph.hpp"


int thread_num=1;


class GcnOptionHelper : public OptionHelper
{
private:
    args::ValueFlag<std::string> input_path_flag;
    args::ValueFlag<std::string> output_path_flag;
    args::ValueFlag<std::string> static_comp_flag;
    args::ValueFlag<vertex_id_t> v_num_flag;
    args::ValueFlag<int> t_num_flag;
    args::Flag make_undirected_flag;
public:
    std::string input_path;
    std::string output_path;
    std::string static_comp;
    bool make_undirected;
    vertex_id_t v_num;
    int t_num;
    GcnOptionHelper() :
        v_num_flag(parser, "vertex", "vertex number", {'v'}),
        t_num_flag(parser, "thread", "thread number", {'t'}),
        input_path_flag(parser, "input", "input graph path", {'i'}),
        output_path_flag(parser, "output", "output graph path", {'o'}),
        static_comp_flag(parser, "static", "graph type: [weighted | unweighted]", {'s'}),
        make_undirected_flag(parser, "make-undirected", "load graph and treat each edge as undirected edge", {"make-undirected"})
    {
    }

    virtual void parse(int argc, char **argv)
    {
        OptionHelper::parse(argc, argv);

        assert(v_num_flag);
        v_num = args::get(v_num_flag);

        assert(t_num_flag);
        t_num = args::get(t_num_flag);

        assert(input_path_flag);
        input_path = args::get(input_path_flag);

        assert(output_path_flag);
        output_path = args::get(output_path_flag);

        assert(static_comp_flag);
        static_comp = args::get(static_comp_flag);
        assert(static_comp.compare("weighted") == 0 || static_comp.compare("unweighted") == 0);

        make_undirected = make_undirected_flag;
    }
};

template<typename edge_data_t, typename ec_data_t>
void build_edge_container(vertex_id_t v_num,Edge<edge_data_t> *edges, edge_id_t local_edge_num, EdgeContainer<ec_data_t> *ec, vector<vertex_id_t>& vertex_out_degree)
{

    ec->adj_lists = new AdjList<ec_data_t>[v_num];

    ec->adj_units = new AdjUnit<ec_data_t>[local_edge_num];
    edge_id_t chunk_edge_idx = 0;
    for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
    {
        ec->adj_lists[v_i].begin = ec->adj_units + chunk_edge_idx;
        ec->adj_lists[v_i].end = ec->adj_lists[v_i].begin;

        chunk_edge_idx += vertex_out_degree[v_i];
    }
    for (edge_id_t e_i = 0; e_i < local_edge_num; e_i++)
    {
        auto e = edges[e_i];
        auto ep = ec->adj_lists[e.src].end ++;

        ep->neighbour = e.dst;
        ep->data = -1;

    }
}

void read_txt_edges(const char* input_path, std::vector<Edge<EmptyData> > &edges)
{
    FILE *f = fopen(input_path, "r");
    assert(f != NULL);
    vertex_id_t src, dst;
    while (2 == fscanf(f, "%u %u", &src, &dst))
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
    while (3 == fscanf(f, "%u %u %d", &src, &dst, &weight))
    {
        edges.push_back(Edge<int>(src, dst, weight));
    }
    fclose(f);
}


template<typename edge_data_t>
void sort_csr(EdgeContainer<edge_data_t>* csr, vertex_id_t v_num) {
#pragma omp parallel for  num_threads(thread_num)
    for (int v_i = 0; v_i < v_num; v_i++) {
        std::sort(csr->adj_lists[v_i].begin, csr->adj_lists[v_i].end,
            [](const AdjUnit<edge_data_t> a, const AdjUnit<edge_data_t> b) {return a.neighbour < b.neighbour; });
    }
}

template<typename T>
AdjUnit<T>* getCN_bs(vertex_id_t src, vertex_id_t dst, EdgeContainer<T>* csr) {
    AdjList<T>* adj_list = &csr->adj_lists[src];
    AdjUnit<T> target;
    target.neighbour = dst;
    auto edge = std::lower_bound(adj_list->begin, adj_list->end, target, [](const AdjUnit<T> & a, const AdjUnit<T> & b) { return a.neighbour < b.neighbour; });

    if (edge != adj_list->end && dst == edge->neighbour)
        return edge;
    else
        return nullptr;
}

int myIntersectition(const std::vector<vertex_id_t>& v1,const std::vector<vertex_id_t>& v2,std::vector<vertex_id_t>& v_intersection)
{
    int cn=0;
    int p1=0;
    int p2=0;
    int v1_sz = v1.size();
    int v2_sz = v2.size();
    const vector<vertex_id_t>*long_v;
    const vector<vertex_id_t>*short_v;
    if(v1_sz>v2_sz){
        long_v=&v1;
        short_v=&v2;
    }else{
        long_v=&v2;
        short_v=&v1;
    }
    int max_sz = max(v1_sz,v2_sz);
    int min_sz = min(v1_sz,v2_sz);
    
    int begin = 0;

    if((*long_v)[max_sz-1]<(*short_v)[0]||((*long_v)[0]>(*short_v)[min_sz-1])||v1.empty()||v2.empty())return 0;
    while(p1<max_sz&&p2<min_sz){
        int offset = 1;
        int last_p = offset;
        if((*short_v)[p2]<((*long_v)[p1])){
            p2++;
            continue;
        }
        while((*long_v)[p1+offset-1]<(*short_v)[p2]){
            offset=offset*2;
            last_p = p1+offset<max_sz?offset:max_sz-p1;
            if(p1+offset>=max_sz)break;
        }
        if((*long_v)[max_sz-1]<(*short_v)[p2]){
            p2++;
            break;
        }
        auto iter = lower_bound(long_v->begin()+(p1+offset/2),long_v->begin()+p1+last_p,(*short_v)[p2]); 
        int t = iter - long_v->begin();
        if(*iter==(*short_v)[p2]){
            v_intersection.push_back((*short_v)[p2]);
            cn++;
            p2++;
            p1=t++;
        }else{
            p2++;
            p1=(p1+offset/2);
        }
    };
    return cn;   
   
}
void interset_debug(vertex_id_t src, vertex_id_t dst,EdgeContainer<int>*graph)
{
    printf(" src dst %d\n",getCN_bs<int>(0,257,graph)->data);
    std::vector<vertex_id_t>v1(graph->adj_lists[src].end - graph->adj_lists[src].begin);
        std::vector<vertex_id_t>v2(graph->adj_lists[dst].end - graph->adj_lists[dst].begin);
        vertex_id_t i=0;
        printf("v1: ");
        for(auto a=graph->adj_lists[src].begin;a< graph->adj_lists[src].end;a++,i++){
            v1[i] = a->neighbour;
            printf("%u ",v1[i]);
        }
        i=0;
        printf("\nv2: ");
        for(auto a=graph->adj_lists[dst].begin;a< graph->adj_lists[dst].end;a++,i++){
            v2[i] = a->neighbour;
            printf("%u ",v2[i]);
        }
        std::vector<vertex_id_t> v_intersection;
         myIntersectition(v1,v2,v_intersection);
         printf("\ncn: ");
         for(auto a: v_intersection){
             printf("%u ",a);
         }
        vertex_id_t cn = v_intersection.size();
        
        if(cn>0)
        {
            if(v_intersection[0] == src || v_intersection[0]== dst)
                cn--;
            
            if(cn>1)
            {
                for(vertex_id_t i = 1; i < v_intersection.size(); i++)
                {
                    if(v_intersection[i]==v_intersection[i-1])
                        cn--;
                    else if(v_intersection[i]==src || v_intersection[i]==dst)
                        cn--;
                }
            }
        }
        printf("\n%u %u cn :%d\n",src,dst,cn);
}

template<typename edge_data_t>
void gcn(const char* input_path, const char* output_path, bool load_as_undirected ,vertex_id_t v_num)
{
    std::vector<Edge<edge_data_t> > edges;
    read_txt_edges(input_path, edges);

    std::vector<Edge<edge_data_t> >* edge_ptr = &edges;
    std::vector<Edge<edge_data_t>> undirected_edges;

     if (load_as_undirected)
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
    printf("%lu \n",vertex_out_degree.size());
   
    for (edge_id_t e_i = 0; e_i < edge_ptr->size(); e_i++) 
    {

        vertex_out_degree[(*edge_ptr)[e_i].src]++;
    }
    printf("cal degree ok\n");
    EdgeContainer<int>*graph = new EdgeContainer<int>();
    build_edge_container<edge_data_t,int>(v_num,(*edge_ptr).data(), (*edge_ptr).size(), graph, vertex_out_degree);
    printf("build csr ok\n");
    sort_csr(graph,v_num);

    // for(int i=0;i<v_num;i++){
    //     printf("%u\n",i);
    //     for(auto a=graph->adj_lists[i].begin;a<graph->adj_lists[i].end;a++){
    //         printf("%u|%d ",a->neighbour,a->data);
    //     }
    //     printf("\n");
    // }

    printf("sort csr ok\n");

    // interset_debug(0,7100,graph);

#pragma omp parallel for num_threads(thread_num)
    for(edge_id_t e_i = 0; e_i < edges.size(); e_i ++){

        vertex_id_t src = edges[e_i].src;
        vertex_id_t dst = edges[e_i].dst;

        auto edge_pos = getCN_bs<int>(src,dst,graph);
        if(edge_pos->data != -1){
            continue;
        }

        std::vector<vertex_id_t>v1(graph->adj_lists[src].end - graph->adj_lists[src].begin);
        std::vector<vertex_id_t>v2(graph->adj_lists[dst].end - graph->adj_lists[dst].begin);
        vertex_id_t i=0;
        for(auto a=graph->adj_lists[src].begin;a< graph->adj_lists[src].end;a++,i++){
            v1[i] = a->neighbour;
        }
        i=0;
        for(auto a=graph->adj_lists[dst].begin;a< graph->adj_lists[dst].end;a++,i++){
            v2[i] = a->neighbour;
        }
        std::vector<vertex_id_t> v_intersection;
        // printf("src %u dst %u v1: %u v2: %u  ",src,dst,v1.size(),v2.size());
        myIntersectition(v1,v2,v_intersection);

        vertex_id_t cn = v_intersection.size();

        if(cn>0)
        {
            if(v_intersection[0] == src || v_intersection[0]== dst)
                cn--;
            
            if(cn>1)
            {
                for(vertex_id_t i = 1; i < v_intersection.size(); i++)
                {
                    if(v_intersection[i]==v_intersection[i-1])
                        cn--;
                    else if(v_intersection[i]==src || v_intersection[i]==dst)
                        cn--;
                }
            }
        }
        
        edge_pos->data = cn;
        
        // printf("cn %d\n",cn);

        auto mirror_edge = getCN_bs<int>(dst,src,graph);
        if(mirror_edge!=nullptr){
            mirror_edge->data = cn;
        }
        
    }

    FILE* output_file = fopen(output_path,"w");
     for(edge_id_t e_i = 0; e_i < edges.size(); e_i ++){

        vertex_id_t src = edges[e_i].src;
        vertex_id_t dst = edges[e_i].dst;

        int cn = getCN_bs<int>(src,dst,graph)->data;

        fprintf(output_file,"%u %u %d\n",src,dst,cn);
     }
    
    fclose(output_file);

    printf("%zu common neighbours are calculated\n", edges.size());
}

int main(int argc, char** argv)
{
    Timer timer;
    srand(time(NULL));
    GcnOptionHelper opt;
    opt.parse(argc, argv);
    thread_num = opt.t_num;
    if(thread_num==0)thread_num=1;
    printf("thread_num: %d\n",thread_num);
   
    gcn<EmptyData>(opt.input_path.c_str(), opt.output_path.c_str(),opt.make_undirected,opt.v_num);


    printf("[get cn] time: %f \n",timer.duration());
	return 0;
}
