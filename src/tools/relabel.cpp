#include <assert.h>

#include <iostream>
#include <utility>
#include <type_traits>
#include <algorithm>

#include "storage.hpp"
#include "type.hpp"
#include "option_helper.hpp"
#include "graph.hpp"




class RelabelOptionHelper : public OptionHelper
{
private:
    args::ValueFlag<std::string> input_path_flag;
    args::ValueFlag<std::string> output_path_flag;
public:
    std::string input_path;
    std::string output_path;
    RelabelOptionHelper(string head,string tail) :
        OptionHelper(head,tail),
        input_path_flag(parser, "input", "input graph path", {'i'}),
        output_path_flag(parser, "output", "output graph path", {'o'})
    {
    }

    virtual void parse(int argc, char **argv)
    {
        OptionHelper::parse(argc, argv);


        assert(input_path_flag);
        input_path = args::get(input_path_flag);

        assert(output_path_flag);
        output_path = args::get(output_path_flag);

    }
};

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

void read_txt_edges(const char* input_path, std::vector<Edge<vertex_id_t> > &edges)
{
    FILE *f = fopen(input_path, "r");
    vertex_id_t src, dst;
    vertex_id_t weight;
    while (3 == fscanf(f, "%u %u %u", &src, &dst, &weight))
    {
        edges.push_back(Edge<vertex_id_t>(src, dst, weight));
    }
    fclose(f);
}


template<typename edge_data_t>
void grelabel(const char* input_path, const char* output_path)
{
    std::vector<Edge<edge_data_t> > edges;
    read_txt_edges(input_path, edges);

    long long max_vid=0;
    edge_id_t max_eid=0;
    for (edge_id_t e_i = 0; e_i < edges.size(); e_i++) 
    {
        vertex_id_t src = edges[e_i].src;
        vertex_id_t dst = edges[e_i].dst;
        if(src>max_vid){
            max_vid=src;
            max_eid = e_i;
        }
        if(dst>max_vid){
            max_vid=dst;
            max_eid = e_i;
        }
    }
    printf("eid: %lu max id: %llu\n",max_eid,max_vid);


    std::vector<vertex_id_t> relabel(max_vid+1);
    std::vector<bool> isRelabel(max_vid+1,false);

    vertex_id_t pos = 0;

    for (edge_id_t e_i = 0; e_i < edges.size(); e_i++) 
    {
        vertex_id_t src = edges[e_i].src;
        vertex_id_t dst = edges[e_i].dst;
        if(isRelabel[src]==false){
            relabel[src]=pos;
            pos++;
            isRelabel[src]=true;
        }
        if(isRelabel[dst]==false){
            relabel[dst]=pos;
            pos++;
            isRelabel[dst]=true;
        }
    }
    printf("relabel ok, start writing\n");
 
    FILE* output_file = fopen(output_path,"w");
    if(output_file==nullptr){
        printf("open %s fail\n",output_path);
    }
     for(edge_id_t e_i = 0; e_i < edges.size(); e_i ++){
        
        vertex_id_t src = edges[e_i].src;
        vertex_id_t dst = edges[e_i].dst;

        // fprintf(output_file,"%u %u %u\n",relabel[src],relabel[dst],edges[e_i].data);
        fprintf(output_file,"%u %u\n",relabel[src],relabel[dst]);
     }
    
    fclose(output_file);

    printf("%zu edges are relabeled \nvertex num: %u\n", edges.size(),pos);
}

int main(int argc, char** argv)
{
    Timer timer;
    RelabelOptionHelper opt("Preprocess the raw graph data.\n\
Relabel the vertax id to continuous natural number, and count the number of vertices","[WARN] Assume the input graph data is undirected and unweighted by default");
    opt.parse(argc, argv);

    grelabel<EmptyData>(opt.input_path.c_str(), opt.output_path.c_str());

    printf("[relabel] time: %f \n",timer.duration());
	return 0;
}
