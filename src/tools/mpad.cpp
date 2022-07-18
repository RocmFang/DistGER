
#include <iostream>
#include<vector>
#include<queue>
#include<stack>
#include<string>
#include <stdio.h>
#include <algorithm>
#include <utility>
#include<ctime>
#include<string>
#include<omp.h>
#include <sys/time.h>
#include<algorithm>
#include<thread>
#include<unistd.h>
#include <unordered_map>
#include <mutex>

#include "type.hpp"
#include "option_helper.hpp"
#include "graph.hpp"


using namespace std;


#define OMP_PARALLEL_THRESHOLD 4000
typedef unsigned long long ulonglong;

int worker_num = max(1, ((int)thread::hardware_concurrency()) - 1);
string stream_mod;

FILE* flog ;
FILE* ftime;

class MpadOptionHelper : public OptionHelper
{
private:
    args::ValueFlag<std::string> train_graph_flag;
    args::ValueFlag<std::string> test_edges_flag;
    args::ValueFlag<int> v_num_flag;
    args::ValueFlag<int> p_num_flag;
public:
    std::string train_graph;
    std::string test_edges;
    int v_num;
    int p_num;
    MpadOptionHelper() :
        train_graph_flag(parser, "train", "train graph path", {'i'}),
        test_edges_flag(parser, "test", "test edges path", {'e'}),
        v_num_flag(parser, "vertex", "graph vertex number", {'v'}),
        p_num_flag(parser, "partition", "partition number", {'p'})
    {
    }

    virtual void parse(int argc, char **argv)
    {
        OptionHelper::parse(argc, argv);

        assert(train_graph_flag);
        train_graph = args::get(train_graph_flag);

        assert(test_edges_flag);
        test_edges = args::get(test_edges_flag);

        assert(v_num_flag);
        v_num = args::get(v_num_flag);

        assert(p_num_flag);
        p_num = args::get(p_num_flag);
    }
};

template<typename T>
void get_degree(Edge<T>* edges, size_t e_num,  vector<vertex_id_t>& degree,vertex_id_t v_num){
	
	degree.resize(v_num);
	for (edge_id_t e_i = 0; e_i < e_num; e_i++)
	{
		degree[edges[e_i].src]++;
	}
}

template<typename T>
void build_edge_container(Edge<T>* edges, edge_id_t e_num, vertex_id_t v_num, EdgeContainer<T>* ec, const vector<vertex_id_t>& vertex_out_degree)
{
	ec->adj_lists = new AdjList<T>[v_num];
	ec->adj_units = new AdjUnit<T>[e_num];
	edge_id_t chunk_edge_idx = 0;
	for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
	{
		ec->adj_lists[v_i].begin = ec->adj_units + chunk_edge_idx;
		ec->adj_lists[v_i].end = ec->adj_lists[v_i].begin;
		chunk_edge_idx += vertex_out_degree[v_i];
	}
	
	for (edge_id_t e_i = 0; e_i < e_num; e_i++)
	{
		auto e = edges[e_i];
		auto ep = ec->adj_lists[e.src].end++; 
		ep->neighbour = e.dst;
		ep->data = e.data;
	}
}

void logtime(const char* type,time_t duration)
{
    time_t sec = duration;
    float min = duration/60.0;
    float hour = min/60.0;
    fprintf(flog,"[%s]%ld s = %f min = %f h\n",type,sec,min,hour);
}


template<typename T>
void sort_csr(EdgeContainer<T>* csr, vertex_id_t v_num) {
    for (int v_i = 0; v_i < v_num; v_i++) {
        std::sort(csr->adj_lists[v_i].begin, csr->adj_lists[v_i].end,
            [](const AdjUnit<T> a, const AdjUnit<T> b) {return a.neighbour < b.neighbour; });
    }
}
void myIntersectition(const vector<vertex_id_t>& v1,const vector<vertex_id_t>& v2,vector<vertex_id_t>& v_intersection)
{
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
    if(v1.empty()||v2.empty())return;
    if((*long_v)[max_sz-1]<(*short_v)[0]||((*long_v)[0]>(*short_v)[min_sz-1]))return;
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
        auto iter = lower_bound(long_v->begin()+(p1+offset/2),long_v->begin()+p1+last_p,(*short_v)[p2]); // 如果在 区间找到了
        int t = iter - long_v->begin();
        if(*iter==(*short_v)[p2]){
            v_intersection.push_back((*short_v)[p2]);
            p2++;
            p1=t++;
        }else{
            p2++;
            p1=(p1+offset/2);
        }
    };   
   
}
// v1 v2 都是有序的

template<typename T>
int intersection(AdjList<T>* v1,const vector<vertex_id_t>& v2,vector<vertex_id_t>& v_intersection, const vector<vertex_id_t>& graph_degree,vertex_id_t sum_degree)
{
    int sz = v1->end - v1->begin;
    vector<vertex_id_t>v1_v;

    for (int i = 0; i < sz; i++)
    {
            if(v1_v.empty()||(v1->begin + i)->neighbour!=(v1->begin + i-1)->neighbour)        
                v1_v.push_back((v1->begin + i)->neighbour); 
    }
    vertex_id_t current_degree = 0;

    myIntersectition(v1_v,v2,v_intersection);

    return v_intersection.size();
}
template<typename T>
T getCN_bs(vertex_id_t src, vertex_id_t dst, EdgeContainer<T>* csr) {
    AdjList<T>* adj_list = &csr->adj_lists[src];
    AdjUnit<T> target;
    target.neighbour = dst;// 上一跳顶点
    auto edge = std::lower_bound(adj_list->begin, adj_list->end, target, [](const AdjUnit<T> & a, const AdjUnit<T> & b) { return a.neighbour < b.neighbour; });
    if (edge != adj_list->end && dst == edge->neighbour)
        return edge->data;
    else
        return 0;
}
template<typename T>
size_t count_cn(const vector<vertex_id_t>& vertex_array, vertex_id_t v, EdgeContainer<T>*cn_csr,int partition_num)
{
    size_t sum_cn = 0;
    bool use_parallel = false;
    int thread_num = worker_num/partition_num-1;
    if(vertex_array.size()>OMP_PARALLEL_THRESHOLD)
        use_parallel = true;
    // #pragma omp parallel for reduction(+:sum_cn) if(use_parallel) num_threads(thread_num)
    for (int i = 0; i < vertex_array.size(); i++)
    {
        sum_cn += getCN_bs(v, vertex_array[i], cn_csr);
    }
    return sum_cn;
}

string str_cb(char const* a,char const* b)
{
    std::string const& cc = std::string(a) + std::string(b);
    char const* c = cc.c_str();
    return c;
}
template<typename T>
void convert_undirecteds(Edge<T>*& read_edges, size_t& read_e_num)
{
    Edge<T>* undirected_edges = new Edge<T>[read_e_num * 2];
    for (size_t e_i = 0; e_i < read_e_num; e_i++)
    {
        // 首先获得读取的边
        undirected_edges[e_i * 2] = read_edges[e_i];
        // 在将边交换
        std::swap(read_edges[e_i].src, read_edges[e_i].dst);
        undirected_edges[e_i * 2 + 1] = read_edges[e_i];
    }
    delete[]read_edges;
    read_edges = undirected_edges;
    read_e_num *= 2; // 更新 边数 
}

template<typename T>
void BFS(EdgeContainer<T>* graph_csr,std::queue<vertex_id_t>& stream,vertex_id_t v_num,vector<vertex_id_t>& graph_degree)
{

      vector<bool>is_visited(v_num,false);
    queue<vertex_id_t> que;
    vertex_id_t max_degree = 0;
    vertex_id_t max_d_i ;
    for(vertex_id_t v_i=0; v_i < v_num;v_i++ )
    {
        if(graph_degree[v_i]>max_degree && is_visited[v_i]==false){
            max_degree = graph_degree[v_i];
            max_d_i = v_i;
        }
    }
    que.push(max_d_i);
    is_visited[max_d_i]=true;
    while(!que.empty())
    {
        vector<vertex_id_t> t;
        vertex_id_t que_front = que.front();
        que.pop();
        stream.push(que_front);
        for(auto a=graph_csr->adj_lists[que_front].begin; a < graph_csr -> adj_lists[que_front].end; a++)
        {
            vertex_id_t ner = a->neighbour;
            if(is_visited[ner] == false){
                t.push_back(ner);
                is_visited[ner] = true;
            }
        }
        for(int i = 0; i< t.size(); i++)
        {
            que.push(t[i]);   
        }
    }
    printf("stream size: %zu  %.2f%%\n",stream.size(),stream.size()/(float)v_num*100);
    for(vertex_id_t v=0;v<v_num;v++){
        if(is_visited[v]==false){
            stream.push(v);
            is_visited[v] = true;
        }
    }
    assert(v_num==stream.size());
}

template<typename T>
void BFS_degree(EdgeContainer<T>* graph_csr,std::queue<vertex_id_t>& stream,vertex_id_t v_num,vector<vertex_id_t>& graph_degree)
{

    vector<bool>is_visited(v_num,false);
    queue<vertex_id_t> que;
    vertex_id_t max_degree = 0;
    vertex_id_t max_d_i ;
    for(vertex_id_t v_i=0; v_i < v_num;v_i++ )
    {
        if(graph_degree[v_i]>max_degree && is_visited[v_i]==false){
            max_degree = graph_degree[v_i];
            max_d_i = v_i;
        }
    }
    que.push(max_d_i);
    is_visited[max_d_i]=true;
    while(!que.empty())
    {
        vector<vertex_id_t> t;
        vertex_id_t que_front = que.front();
        que.pop();
        stream.push(que_front);
        for(auto a=graph_csr->adj_lists[que_front].begin; a < graph_csr -> adj_lists[que_front].end; a++)
        {
            vertex_id_t ner = a->neighbour;
            if(is_visited[ner] == false){
                t.push_back(ner);
                is_visited[ner] = true;
            }
        }
        sort(t.begin(),t.end(),[&](vertex_id_t a,vertex_id_t b){
            return graph_degree[a] > graph_degree[b];
        });
        for(int i = 0; i< t.size(); i++)
        {
            que.push(t[i]);   
        }
    }
    printf("stream size: %zu    %.2f%%\n",stream.size(),stream.size()/(float)v_num*100);
      for(vertex_id_t v=0;v<v_num;v++){
        if(is_visited[v]==false){
            stream.push(v);
            is_visited[v] = true;
        }
    }
    assert(v_num==stream.size());
}

template<typename T>
void DFS(EdgeContainer<T>* graph_csr,std::queue<vertex_id_t>& stream,vertex_id_t v_num,vector<vertex_id_t>& graph_degree)
{

    vector<bool>is_visited(v_num,false);
    stack<vertex_id_t> sta;
    vertex_id_t max_degree = 0;
    vertex_id_t max_d_i ;
    for(vertex_id_t v_i=0; v_i < v_num;v_i++ )
    {
        if(graph_degree[v_i]>max_degree && is_visited[v_i]==false){
            max_degree = graph_degree[v_i];
            max_d_i = v_i;
        }
    }
    sta.push(max_d_i);
    is_visited[max_d_i]=true;
    while(!sta.empty())
    {
        vertex_id_t sta_top = sta.top();
        stream.push(sta_top);
        sta.pop();
        vector<vertex_id_t> t;
        for(auto a =  graph_csr->adj_lists[sta_top].begin;a < graph_csr->adj_lists[sta_top].end; a++)
        {
            if( is_visited[a->neighbour]== false){
                t.push_back(a->neighbour);
                is_visited[a->neighbour] = true;
            }
        }
        for(int i =0;i< t.size();i++)
        {
            sta.push(t[i]);
        }
    }
     printf("stream size: %zu    %.2f%%\n",stream.size(),stream.size()/(float)v_num*100);
      for(vertex_id_t v=0;v<v_num;v++){
        if(is_visited[v]==false){
            stream.push(v);
            is_visited[v] = true;
        }
    }
    assert(v_num==stream.size());
}

template<typename T>
void DFS_degree(EdgeContainer<T>* graph_csr,std::queue<vertex_id_t>& stream,vertex_id_t v_num,vector<vertex_id_t>& graph_degree)
{

   vector<bool>is_visited(v_num,false);
    stack<vertex_id_t> sta;
    vertex_id_t max_degree = 0;
    vertex_id_t max_d_i ;
    for(vertex_id_t v_i=0; v_i < v_num;v_i++ )
    {
        if(graph_degree[v_i]>max_degree && is_visited[v_i]==false){
            max_degree = graph_degree[v_i];
            max_d_i = v_i;
        }
    }
    sta.push(max_d_i);
    is_visited[max_d_i]=true;
    while(!sta.empty())
    {
        vertex_id_t sta_top = sta.top();
        stream.push(sta_top);
        sta.pop();
        vector<vertex_id_t> t;
        for(auto a =  graph_csr->adj_lists[sta_top].begin;a < graph_csr->adj_lists[sta_top].end; a++)
        {
            if( is_visited[a->neighbour]== false){
                t.push_back(a->neighbour);
                is_visited[a->neighbour] = true;
            }
        }
        sort(t.begin(),t.end(),[&](vertex_id_t a,vertex_id_t b){
            return graph_degree[a] < graph_degree[b];
        });
        for(int i =0;i< t.size();i++)
        {
            sta.push(t[i]);
        }
    }
    printf("stream size: %zu    %.2f%%\n",stream.size(),stream.size()/(float)v_num*100);
      for(vertex_id_t v=0;v<v_num;v++){
        if(is_visited[v]==false){
            stream.push(v);
            is_visited[v] = true;
        }
    }
    
    assert(v_num==stream.size());
}

template<typename T>
void vertex_partition_LDG_sum(EdgeContainer<T>* graph_csr,vertex_id_t v_num, vector<std::vector<vertex_id_t>>& partition_vertex_array, float beta,  vector<vertex_id_t>& graph_degree)
{
 
    int partition_num = partition_vertex_array.size();
    
               
    std::vector<float> partition_common_score(partition_num,0.0);//当前点和机器已有点的关联度得分，把点放到得分最高的机器
    std::vector<vertex_id_t> partition_sum_degree(partition_num,0);
    std::queue<vertex_id_t> q;// vertex queue
    size_t current_v_num = 0;

    // sort by degress
    // Timer timer_srt;
    // std::vector<pair<vertex_id_t,vertex_id_t>>idx_degree(v_num);
    // for (vertex_id_t v_i = 0; v_i < v_num; v_i++) {
    //     idx_degree[v_i] = {v_i,graph_degree[v_i]};
    // }
    // printf("[idx_degree init]%lf s\n",timer_srt.duration());
    // timer_srt.restart();
    // sort(idx_degree.begin(),idx_degree.end(),[](pair<vertex_id_t,vertex_id_t>&a,pair<vertex_id_t,vertex_id_t>&b){return a.second>b.second;});
    // printf("[idx_degree sort]%lf s\n",timer_srt.duration());

    // assert(idx_degree.size()>0);
    // vertex_id_t puted_num = 0;// the vertices num have been puted
    // vertex_id_t init_workload_per = idx_degree[0].second;// the largest degree
    // vector<vertex_id_t>init_max;// record the vertex have been puted
    // for(int p_i = 0; p_i < partition_num; p_i++)
    // {
    //     vertex_id_t work_load = 0;
    //     int dist_add_before = init_workload_per-work_load;
    //     dist_add_before = abs(dist_add_before);
    //     int dist_add_after = idx_degree[puted_num].second+work_load-init_workload_per;
    //     dist_add_after = abs(dist_add_after);
    //     while(dist_add_before >= dist_add_after){
    //         partition_vertex_array[p_i].push_back(idx_degree[puted_num].first);
    //         partition_sum_degree[p_i]+=idx_degree[puted_num].second; // init partition sum degree
    //         init_max.push_back(idx_degree[puted_num].first);
    //         work_load += idx_degree[puted_num].second;
    //         puted_num ++ ;
    //         dist_add_before = init_workload_per-work_load;
    //         dist_add_before = abs(dist_add_before);
    //         dist_add_after = idx_degree[puted_num].second+work_load-init_workload_per;
    //         dist_add_after = abs(dist_add_after);
    //     }		
    // }
    // fprintf(flog,"puted num %u \n",puted_num);
    // printf("puted num %u \n",puted_num);
    // assert(puted_num<=v_num);
    // current_v_num = puted_num;
    // for (int i = 0; i < partition_num; i++)
    // {
    //     sort(partition_vertex_array[i].begin(),partition_vertex_array[i].end());
    // };
    
    // for (vertex_id_t v_i = current_v_num; v_i < v_num; v_i++) {// 初始化q
    //          q.push(idx_degree[v_i].first);	
    // }

    


    //random
    // vector<vertex_id_t> ran_arr(v_num);
    // for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
    // {
    //     ran_arr[v_i] = v_i;
    // }
    // ulonglong next_random = 5;
    // for(vertex_id_t i=0;i<v_num;i++)
    // {
    //     next_random = next_random * (ulonglong)25214903917 + 11;
    //     vertex_id_t v1 = next_random%v_num;
    //     next_random = next_random * (ulonglong)25214903917 + 11;
    //     vertex_id_t v2 = next_random%v_num;
    //     std::swap(ran_arr[v1],ran_arr[v2]);
    // }
    //  for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
    // {
    //     q.push(ran_arr[v_i]);
    // }

    stream_mod="-DFS-degree";
    cout<<"stream mode: "<<stream_mod<<endl;
    // BFS(graph_csr,q,v_num,graph_degree);
    // BFS_degree(graph_csr,q,v_num,graph_degree);
    // DFS(graph_csr,q,v_num,graph_degree);
    DFS_degree(graph_csr,q,v_num,graph_degree);

    time_t pa_s = time(NULL);
    Timer partition_timer;
    omp_set_dynamic(0);
    omp_set_nested(true); 
    vertex_id_t iter = 0;
    size_t last_v  =0;
    while (!q.empty())
    {
        if(current_v_num - last_v > 1000){
            printf("\rProgress: %.2f%%",current_v_num/(float)v_num*100);
            last_v = current_v_num;
        }
        iter++;

        float c = beta * current_v_num / partition_num;
        vertex_id_t v_i = q.front();
        q.pop();
        std::fill(partition_common_score.begin(), partition_common_score.end(), 0.0);	
        
        #pragma omp parallel for num_threads(partition_num)
        for (int p_i = 0; p_i < partition_num; p_i++)// 先计算 cn 得分
        {
            vector<vertex_id_t> v_intersection;
            int nb_score;
            
            nb_score = intersection(&graph_csr->adj_lists[v_i], partition_vertex_array[p_i],v_intersection, graph_degree,partition_sum_degree[p_i]);

            int cn_score; 

            cn_score = count_cn(v_intersection, v_i, graph_csr,partition_num);

            float w =1- partition_vertex_array[p_i].size() / c;
            partition_common_score[p_i] = ((float)cn_score +(float) nb_score + 1)*w;
    
        }

        int maxp = max_element(partition_common_score.begin(), partition_common_score.end()) - partition_common_score.begin();
        partition_vertex_array[maxp].insert(lower_bound(partition_vertex_array[maxp].begin(), partition_vertex_array[maxp].end(), v_i), v_i);
        partition_sum_degree[maxp]+=graph_degree[v_i];
        current_v_num++;
       
    }

    time_t pa_e = time(NULL);

    fprintf(flog,"[sum partition time] %lf s (%lf h)\n",partition_timer.duration(),partition_timer.duration()/60.0/60.0);
    
    logtime("sum pa time",pa_e-pa_s);
    
    assert(current_v_num==v_num);
}


template<typename T>
void read_edge_by_txt(const char* fname, Edge<T>*& edge, size_t& e_num)
{
	FILE * f = fopen(fname, "r");
	assert(f != NULL);

	fseek(f, 0, SEEK_END);
    size_t end = ftell(f);
	Edge<T> temp;
	e_num = 0;
	fseek(f, 0, SEEK_SET);
	size_t ft = ftell(f);
	while (ft < end)
	{
		if (read_edge_txt(f, &temp))
		{
			e_num++;
			// printf("\r%zu",e_num);
		}
		ft = ftell(f);
	}
	edge = new Edge<T>[e_num];

	size_t e_i = 0;
	fseek(f, 0, SEEK_SET);
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

template<typename T>
void partition_relabel(const char* graph_path,const char* test_edges_path, vertex_id_t v_num,int partition_num)
{
    
    // 1. 读边数据
    Timer timer;
    Edge<T>* graph_edges;
    size_t graph_e_num;
    read_edge_by_txt(graph_path, graph_edges, graph_e_num);

    Edge<EmptyData>* test_edges;
    size_t test_e_num;
    read_edge_by_txt(test_edges_path, test_edges, test_e_num);
    // printf("[test_e_num]%lu\n",test_e_num);
    fprintf(flog,"[read edge time]%lf s\n",timer.duration());
    timer.restart();
    cout << "read graph and cn finish\n";
    // 2. 单边数据转双边
    convert_undirecteds<T>(graph_edges, graph_e_num);
    fprintf(flog,"[convert time]%lf s\n",timer.duration());
    timer.restart();
    cout << "convert_undirecteds finished\n";
    // 3. 根据边数据统计点的度数
    vector<vertex_id_t>graph_degree;
    get_degree(graph_edges, graph_e_num, graph_degree, v_num);
    fprintf(flog,"[count degree time]%lf s\n",timer.duration());
    cout << "count_degree finished\n";
    for(vertex_id_t v =0; v<v_num;v++){
        assert(graph_degree[v]>0);
    }
    timer.restart();
    // 4. 根据 edges 和 degree 构建 csr 数据结构
    EdgeContainer<T>* graph_csr = new EdgeContainer<T>();
    build_edge_container(graph_edges, graph_e_num, v_num, graph_csr, graph_degree);
    
    cout << "build csr finshed\n";
    fprintf(flog,"[build container time]%lf s\n",timer.duration()); 
    timer.restart();
    
    // 5. 把 每个顶点的邻接表按序号排序
    sort_csr(graph_csr, v_num);

    fprintf(flog,"[sort csr time]%lf s\n",timer.duration());
    timer.restart();
    cout << "sort csr finshed\n";	

    // 6. 进行点的划分

    float beta = 2.0;//----------------------------------- 

    vector<vector<vertex_id_t>>partition_vertex_array(partition_num);// 存放划分的结果集

    vertex_partition_LDG_sum(graph_csr,v_num, partition_vertex_array, beta,graph_degree);

    cout << "partition finished\n";
    // 7. 把划分好的点向量转成 KnightKing 中的形式，用一个向量和分割数组表示
    vertex_id_t* vertex_array = new vertex_id_t[v_num]; // 从 new -> old 
    vertex_id_t* vertex_partition_begin = new vertex_id_t[partition_num];
    vertex_id_t* vertex_partition_end = new vertex_id_t[partition_num];
    vertex_id_t v_a_p = 0;
    for (int p_i = 0; p_i < partition_num; p_i++) {
        if (p_i == 0)
        {
            vertex_partition_begin[p_i] = 0;
        }
        else
        {
            vertex_partition_begin[p_i] = vertex_partition_end[p_i - 1];
        }
        vertex_partition_end[p_i] = vertex_partition_begin[p_i] + partition_vertex_array[p_i].size();
        for (vertex_id_t v_i = 0; v_i < partition_vertex_array[p_i].size(); v_i++) {
            vertex_array[v_a_p] = partition_vertex_array[p_i][v_i];
            v_a_p++;
        }
    }
    cout << "partition process\n";
    // 8. 把点的序号 relabel。映射为其在数组中的下标, 把 edges 中都修改好
    vertex_id_t* vertex_map = new vertex_id_t[v_num];// 映射数组，从 old -> new 
    for (vertex_id_t v_i = 0; v_i < v_num; v_i++)
    {
        vertex_map[vertex_array[v_i]] = v_i;
    }
    
    for (edge_id_t e_i = 0; e_i < graph_e_num; e_i++)
    {
        graph_edges[e_i].src = vertex_map[graph_edges[e_i].src];
        graph_edges[e_i].dst = vertex_map[graph_edges[e_i].dst];
    }

    for (edge_id_t e_i=0; e_i < test_e_num; e_i++)
    {
        test_edges[e_i].src = vertex_map[test_edges[e_i].src];
        test_edges[e_i].dst = vertex_map[test_edges[e_i].dst];
    }

    cout << "relabel finished\n";
    
    timer.restart();

    // 9. 把relabel 的图 和 relabel 的cn ，分割方式 输出
    string pn_suffix = to_string(partition_num);
    pn_suffix = "-" + pn_suffix;

    string graph_path_r = str_cb(graph_path, pn_suffix.c_str());
    graph_path_r += stream_mod;
    graph_path_r = str_cb(graph_path_r.c_str(), "-r");

    FILE* f_graph_path_r = fopen(graph_path_r.c_str(), "w");
    for (edge_id_t e_i = 0; e_i < graph_e_num; e_i++)
    {
        if(e_i%2==0)
        fprintf(f_graph_path_r,"%u %u %u\n", graph_edges[e_i].src, graph_edges[e_i].dst, graph_edges[e_i].data);
    }
    fclose(f_graph_path_r);

    string test_path_r = str_cb(test_edges_path,pn_suffix.c_str() );
    test_path_r += stream_mod;
    test_path_r = str_cb(test_path_r.c_str(), "-r");
    FILE* f_test_path_r = fopen(test_path_r.c_str(), "w");
    for (edge_id_t e_i = 0; e_i < test_e_num; e_i++)
    {
        fprintf(f_test_path_r,"%u %u\n", test_edges[e_i].src, test_edges[e_i].dst);
    }
    fclose(f_test_path_r);

    string partition_ = str_cb(graph_path, pn_suffix.c_str());
    partition_ += stream_mod;
    partition_ = str_cb(partition_.c_str(), "-p");
    FILE* f_partition_r = fopen(partition_.c_str(), "w");
    for (int p_i = 0; p_i < partition_num; p_i++)
    {
        fprintf(f_partition_r, "%u %u\n", vertex_partition_begin[p_i], vertex_partition_end[p_i]);
    }
    for(int p_i = 0; p_i < partition_num; p_i++){
        fprintf(f_partition_r, "%u ",vertex_partition_end[p_i]-vertex_partition_begin[p_i]);
    }
    cout << "print finished\n";
    fprintf(flog,"[output time]%lf s\n",timer.duration());
    timer.restart();
    fclose(f_partition_r);
    delete[] graph_csr->adj_lists;
    delete[] graph_csr->adj_units;
    delete[] vertex_array;
    delete[] vertex_map;
    delete[] vertex_partition_begin;
    delete[] vertex_partition_end;
}


int main(int argc, char* argv[])
{
  
    Timer timer;
    MpadOptionHelper opt;
    opt.parse(argc, argv);
    // char* graph_path = opt.train_graph.c_str();
    // char* test_edges_path = opt.test_edges.c_str();
    vertex_id_t v_num = opt.v_num;
    int partition_num = opt.p_num;

    assert(v_num>0&&partition_num>0);

    string partition_log_path = str_cb(opt.train_graph.c_str(), ".log");
    flog = fopen(partition_log_path.c_str(), "w");
    time_t timesec;
    struct tm *t;   
     
    time(&timesec);
    t = localtime(&timesec);
    struct tm t1(*t);
    fprintf(flog,"[start]%d-%d-%d %d:%d:%d\n", 1900+t->tm_year, 1+t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    printf("[start]%d-%d-%d %d:%d:%d\n", 1900+t->tm_year, 1+t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fprintf(flog,"graph path: %s\nv num: %u\n",opt.train_graph.c_str(),v_num);
    printf("graph path: %s\nv num: %u\n",opt.train_graph.c_str(),v_num);

    printf("[Notice]: data type of cn is integer!\n");
    // look out the type of cn
    partition_relabel<uint32_t>(opt.train_graph.c_str(),opt.test_edges.c_str(), v_num,partition_num);

    // partition_relabel<float>(opt.train_graph.c_str(),opt.test_edges.c_str(), v_num,partition_num);

    fprintf(flog,"[timer:runtime] %lf s(%lf h)\n",timer.duration(),timer.duration()/60.0/24.0);
    printf("[timer:runtime] %lf s(%lf h)\n",timer.duration(),timer.duration()/60.0/24.0);
    
       
    time_t timesec2;
    struct tm *t2;
    time(&timesec2);
    t2 = localtime(&timesec2);
     
    fprintf(flog,"[end]%d-%d-%d %d:%d:%d\n", 1900+t2->tm_year, 1+t2->tm_mon, t2->tm_mday, t2->tm_hour, t2->tm_min, t2->tm_sec);  
    printf("[end]%d-%d-%d %d:%d:%d\n", 1900+t2->tm_year, 1+t2->tm_mon, t2->tm_mday, t2->tm_hour, t2->tm_min, t2->tm_sec);      
    int sec = t2->tm_sec-t1.tm_sec;
    int min = t2->tm_min-t1.tm_min;
    int hour =  t2->tm_hour-t1.tm_hour;
    int day = t2->tm_mday-t1.tm_mday;
    if(sec<0){
        sec += 60;
        min--;
    }
    if(min<0){
        min+=60;
        hour--;
    }
    if(hour<0){
        hour += 24;
        day--;
    }
    fprintf(flog,"[duration]%dd %dh %dm %ds\n",day,hour,min,sec);
    printf("[duration]%dd %dh %dm %ds\n",day,hour,min,sec);
    time_t duration = timesec2 - timesec;
    logtime("duration",duration);
    fclose(flog);  
    
}

