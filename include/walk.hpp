
#pragma once

#include "type.hpp"
#include "graph.hpp"
#include "path.hpp"

#include <numeric>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <queue>

using namespace std; 

using precision_t = double;
using frequency_t = int;

// #define minLength 20
const double R = 0.999;
// #define init_round 5
#define delta_R 0.001

#define UNIT_TEST true

template<typename walker_data_t>
struct Walker
{
public:
    walker_id_t id;
    step_t step;
    walker_data_t data;
    double Hn; 
    double Sn; 
    double EnH; 
    double EnS; 
    double EnHS; 
    double EnHH;
    double EnSS; 
    double R; 
    double N; 
    vertex_id_t last_vertex;
    bool traceFlag = false;
    
    void setData(double _Hn, double _Sn, double _EnH, double _EnS, double _EnHS, double _EnHH, double _EnSS, double _R, double _N)
    {
        Hn = _Hn;
        Sn = _Sn;
        EnH = _EnH;
        EnS = _EnS;
        EnHS = _EnHS;
        EnHH = _EnHH;
        EnSS = _EnSS;
        R = _R;
        N = _N;
    }
};

template<>
struct Walker<EmptyData>
{
public:
    uint32_t id;
    union
    {
        step_t step;
        EmptyData data;
    };
};

template<typename edge_data_t>
struct AliasBucket
{
    real_t p;
    AdjUnit<edge_data_t> *p_ptr, *q_ptr;
};

template<typename edge_data_t>
struct AliasTableContainer
{
    AliasBucket<edge_data_t> *buckets;
    AliasBucket<edge_data_t> **index;
    AliasTableContainer() : buckets(nullptr), index(nullptr) {}
    ~AliasTableContainer()
    {
    }
};

class WalkConfig
{
public:
    bool output_file_flag;
    std::string output_path_prefix;
    bool print_with_head_info;
    bool output_consumer_flag;
    std::function<void (PathSet*)> output_consumer_func;
    double rate;

    WalkConfig()
    {
        output_file_flag = false;
        print_with_head_info = false;
        output_consumer_flag = false;
        output_consumer_func = nullptr;
        rate = 1;
    }

    void set_output_file(const char* output_path_prefix_param, bool with_head_info = false)
    {
        assert(output_path_prefix_param != nullptr);
        this->output_file_flag = true;
        this->output_path_prefix = std::string(output_path_prefix_param);
        this->print_with_head_info = with_head_info;
    }

    void set_output_consumer(std::function<void (PathSet*)> output_consumer_func_param)
    {
        assert(output_consumer_func_param != nullptr);
        this->output_consumer_flag = true;
        this->output_consumer_func = output_consumer_func_param;
    }

    void set_walk_rate(double rate_param)
    {
        assert(rate_param > 0 && rate_param <= 1);
        this->rate = rate_param;
    }
};

template<typename edge_data_t, typename walker_data_t>
class WalkerConfig
{
public:
    // walker setting
    walker_id_t walker_num;
    std::function<vertex_id_t (walker_id_t)> walker_init_dist_func;
    std::function<void (Walker<walker_data_t>&, vertex_id_t)> walker_init_state_func;
    std::function<void (Walker<walker_data_t>&, vertex_id_t, AdjUnit<edge_data_t> *)> walker_update_state_func;

    WalkerConfig()
    {
        walker_num = 0;
        walker_init_dist_func = nullptr;
        walker_init_state_func = nullptr;
        walker_update_state_func = nullptr;
    }

    WalkerConfig(
        walker_id_t walker_num_param,
        std::function<void (Walker<walker_data_t>&, vertex_id_t)> walker_init_state_func_param = nullptr,
        std::function<void (Walker<walker_data_t>&, vertex_id_t, AdjUnit<edge_data_t> *)> walker_update_state_func_param = nullptr,
        std::function<vertex_id_t (walker_id_t)> walker_init_dist_func_param = nullptr
    )
    {
        WalkerConfig();
        set_walkers(
            walker_num_param,
            walker_init_state_func_param,
            walker_update_state_func_param,
            walker_init_dist_func_param
        );
    }

    void set_walkers(
        walker_id_t walker_num_param,
        std::function<void (Walker<walker_data_t>&, vertex_id_t)> walker_init_state_func_param = nullptr,
        std::function<void (Walker<walker_data_t>&, vertex_id_t, AdjUnit<edge_data_t> *)> walker_update_state_func_param = nullptr,
        std::function<vertex_id_t (walker_id_t)> walker_init_dist_func_param = nullptr
    )
    {
        this->walker_num = walker_num_param;
        this->walker_init_state_func = walker_init_state_func_param;
        this->walker_update_state_func = walker_update_state_func_param;
        this->walker_init_dist_func = walker_init_dist_func_param;
    }
};

template<typename edge_data_t, typename walker_data_t>
class TransitionConfig
{
public:
	// random walk setting
	std::function<real_t (Walker<walker_data_t>&, vertex_id_t)> extension_comp_func;
	std::function<real_t(vertex_id_t, AdjUnit<edge_data_t>*)> static_comp_func;
	std::function<real_t (Walker<walker_data_t>&, vertex_id_t, AdjUnit<edge_data_t> *)> dynamic_comp_func;
	std::function<real_t(vertex_id_t, AdjList<edge_data_t>*)> dcomp_upperbound_func;
	std::function<real_t(vertex_id_t, AdjList<edge_data_t>*)> dcomp_lowerbound_func;
	std::function<void(Walker<walker_data_t>&, vertex_id_t, AdjList<edge_data_t>*, real_t&, vertex_id_t&)> outlier_upperbound_func;
	std::function<AdjUnit<edge_data_t>*(Walker<walker_data_t>&, vertex_id_t, AdjList<edge_data_t>*, vertex_id_t)> outlier_search_func;

    TransitionConfig()
    {
        extension_comp_func = nullptr;
        static_comp_func = nullptr;
        dynamic_comp_func = nullptr;
        dcomp_upperbound_func = nullptr;
        dcomp_lowerbound_func = nullptr;
        outlier_upperbound_func = nullptr;
        outlier_search_func = nullptr;
    }

    TransitionConfig(
        std::function<real_t (Walker<walker_data_t>&, vertex_id_t)> extension_comp_func_param,
        std::function<real_t(vertex_id_t, AdjUnit<edge_data_t>*)> static_comp_func_param = nullptr,
        std::function<real_t (Walker<walker_data_t>&, vertex_id_t, AdjUnit<edge_data_t> *)> dynamic_comp_func_param = nullptr,
        std::function<real_t(vertex_id_t, AdjList<edge_data_t>*)> dcomp_upperbound_func_param = nullptr,
        std::function<real_t(vertex_id_t, AdjList<edge_data_t>*)> dcomp_lowerbound_func_param = nullptr,
        std::function<void(Walker<walker_data_t>&, vertex_id_t, AdjList<edge_data_t>*, real_t&, vertex_id_t&)> outlier_upperbound_func_param = nullptr,
        std::function<AdjUnit<edge_data_t>*(Walker<walker_data_t>&, vertex_id_t, AdjList<edge_data_t>*, vertex_id_t)> outlier_search_func_param = nullptr
    )
    {
        TransitionConfig();
        set_transition(
            extension_comp_func_param,
            static_comp_func_param,
            dynamic_comp_func_param,
            dcomp_upperbound_func_param,
            dcomp_lowerbound_func_param,
            outlier_upperbound_func_param,
            outlier_search_func_param
        );
    }

    void set_transition(
        std::function<real_t (Walker<walker_data_t>&, vertex_id_t)> extension_comp_func_param,
        std::function<real_t(vertex_id_t, AdjUnit<edge_data_t>*)> static_comp_func_param = nullptr,
        std::function<real_t (Walker<walker_data_t>&, vertex_id_t, AdjUnit<edge_data_t> *)> dynamic_comp_func_param = nullptr,
        std::function<real_t(vertex_id_t, AdjList<edge_data_t>*)> dcomp_upperbound_func_param = nullptr,
        std::function<real_t(vertex_id_t, AdjList<edge_data_t>*)> dcomp_lowerbound_func_param = nullptr,
        std::function<void(Walker<walker_data_t>&, vertex_id_t, AdjList<edge_data_t>*, real_t&, vertex_id_t&)> outlier_upperbound_func_param = nullptr,
        std::function<AdjUnit<edge_data_t>*(Walker<walker_data_t>&, vertex_id_t, AdjList<edge_data_t>*, vertex_id_t)> outlier_search_func_param = nullptr
    )
    {
        if (dynamic_comp_func_param != nullptr)
        {
            assert(dcomp_upperbound_func_param != nullptr);
        } else
        {
            assert(dcomp_upperbound_func_param == nullptr);
            assert(dcomp_lowerbound_func_param == nullptr);
            assert(outlier_upperbound_func_param == nullptr);
            assert(outlier_search_func_param == nullptr);
        }
        assert(outlier_upperbound_func_param == nullptr && outlier_search_func_param == nullptr || outlier_upperbound_func_param != nullptr &&  outlier_search_func_param != nullptr);

        this->extension_comp_func = extension_comp_func_param;
        this->static_comp_func = static_comp_func_param;
        this->dynamic_comp_func = dynamic_comp_func_param;
        this->dcomp_upperbound_func = dcomp_upperbound_func_param;
        this->dcomp_lowerbound_func = dcomp_lowerbound_func_param;
        this->outlier_upperbound_func = outlier_upperbound_func_param;
        this->outlier_search_func = outlier_search_func_param;
    }
};

class Correlation {
public:
	Correlation(precision_t H_0, precision_t S_0, precision_t EH_0, precision_t ES_0,
		precision_t EHS_0, precision_t EHH_0, precision_t ESS_0,
		precision_t N_0, precision_t R_0)
		: _M_Hn(H_0), _M_Sn(S_0), _M_EnH(EH_0), _M_EnS(ES_0), _M_EnHS(EHS_0),
		_M_EnHH(EHH_0), _M_EnSS(ESS_0), _M_N(N_0), _M_R(R_0) { }

	Correlation(){}

    void SetProperty(precision_t H_0, precision_t S_0, precision_t EH_0, precision_t ES_0,
		precision_t EHS_0, precision_t EHH_0, precision_t ESS_0,
		precision_t N_0, precision_t R_0){
            setHn(H_0);
            setSn(S_0);
            setEnH(EH_0);
            setEnS(ES_0);
            setEnHS(EHS_0);
            setEnHH(EHH_0);
            setEnSS(ESS_0);
            setN(N_0);
            setR(R_0);
        }

	static precision_t SnSn_div_Snp1Snp1(precision_t Sn, precision_t step_k) {
		precision_t Snp1 = Sn + step_k;
        double res = std::pow(Sn / Snp1, Sn) * std::pow(precision_t(1) / Snp1, step_k);
        assert(res > 0);
		return res;
	}

	static precision_t FjFj_div_FiFi(precision_t fi, precision_t step_k) {
        assert(fi > 0);
		precision_t fj = fi + step_k;
		return std::pow(fj / fi, fi) * std::pow(fj, step_k);
	}

	static precision_t iterateHn(precision_t Hn, precision_t Sn, precision_t fi, precision_t step_k, int worker_id = 1) {
        // std::cout << "[iterateHn()] " << " worker_id = " << worker_id << " Hn = " << Hn << " Sn = " << Sn << std::endl;
        assert(Sn > 0);
		precision_t t = fi == frequency_t(0) ? std::log2(SnSn_div_Snp1Snp1(Sn, step_k))
			: std::log2(SnSn_div_Snp1Snp1(Sn, step_k) * FjFj_div_FiFi(fi, step_k));
        double res = (Sn * Hn - t) / (Sn + step_k);
		return res;
	}


	///****************************************************************************
	/// @brief iterateEX :
	///****************************************************************************
	static precision_t iterateEX(precision_t EnX, precision_t Xnp1, precision_t Np1) {
		return EnX + (Xnp1 - EnX) / Np1;
	}

	///****************************************************************************
	/// @brief  iterate 
	///****************************************************************************
	void iterate(precision_t fi, precision_t step_k = precision_t(1), int worker_id = 0) {

		precision_t Hnp1 = iterateHn(_M_Hn, _M_Sn, fi, step_k, worker_id);
        if(Hnp1 <= 0)
        {
            std::cout << "[work_ider " << worker_id << "] : [ Hnp1 " << Hnp1 << " _M_Hn = " << _M_Hn << " _M_Sn = " << _M_Sn << " fi " << fi << " step_k " << step_k << " ] "<< std::endl;
        }
        assert(Hnp1 > 0);
		precision_t Snp1 = _M_Sn + step_k;
		precision_t Np1 = _M_N + precision_t(1);

		precision_t Enp1H = iterateEX(_M_EnH, Hnp1, Np1);
		precision_t Enp1S = iterateEX(_M_EnS, Snp1, Np1);
		precision_t Enp1HS = iterateEX(_M_EnHS, Hnp1 * Snp1, Np1);
		precision_t Enp1HH = iterateEX(_M_EnHH, Hnp1 * Hnp1, Np1);
		precision_t Enp1SS = iterateEX(_M_EnSS, Snp1 * Snp1, Np1);

		precision_t Dnp1H = Enp1HH - Enp1H * Enp1H;
        // if(Dnp1H <= 0)
        // {
        //     std::cout << "[iterate] " << " Enp1HH = " << Enp1HH << " Enp1H = " << Enp1H << std::endl;
        // }
        assert(Dnp1H > 0);
		precision_t Dnp1S = Enp1SS - Enp1S * Enp1S;
        assert(Dnp1S > 0);

		precision_t exp = Enp1HS - Enp1H * Enp1S;
		_M_R = exp == precision_t(0) ? precision_t(0) : exp / std::sqrt(Dnp1H * Dnp1S);

		_M_Hn = Hnp1;
		_M_Sn = Snp1;
		_M_EnH = Enp1H;
		_M_EnS = Enp1S;
		_M_EnHS = Enp1HS;
		_M_EnHH = Enp1HH;
		_M_EnSS = Enp1SS;
		_M_N = Np1;
	}

	//read only
	precision_t Hn() { return _M_Hn; }
	precision_t Sn() { return _M_Sn; }
	precision_t EnH() { return _M_EnH; }
	precision_t EnS() { return _M_EnS; }
	precision_t EnHS() { return _M_EnHS; }
	precision_t EnHH() { return _M_EnHH; }
	precision_t EnSS() { return _M_EnSS; }
	precision_t N() { return _M_N; }
	precision_t R() { return _M_R; }

    void setHn(precision_t Hn) { _M_Hn = Hn; }
	void setSn(precision_t Sn) { _M_Sn = Sn; }
	void setEnH(precision_t EnH) { _M_EnH = EnH; }
	void setEnS(precision_t EnS) {  _M_EnS = EnS; }
	void setEnHS(precision_t EnHS) {  _M_EnHS = EnHS; }
	void setEnHH(precision_t EnHH) {  _M_EnHH = EnHH; }
	void setEnSS(precision_t EnSS) {  _M_EnSS = EnSS; }
	void setN(precision_t N) {  _M_N = N; }
	void setR(precision_t R) {  _M_R = R; }

private:
	precision_t _M_Hn;
	precision_t _M_Sn;
	precision_t _M_EnH;
	precision_t _M_EnS;
	precision_t _M_EnHS;
	precision_t _M_EnHH;
	precision_t _M_EnSS;
	precision_t _M_N;
	precision_t _M_R;
};

template<typename edge_data_t, typename walker_data_t>
class WalkEngine : public GraphEngine<edge_data_t>
{
    StdRandNumGenerator* randgen;
    Timer *timer;
    size_t cross_num = 0;
    size_t intr_num = 0;
    size_t trace_num = 0;
    size_t p_step = 0;
    // Timer *timer;

public:
    vector<int>vertex_cn;
    vector<int>local_corpus;
    vector<vertex_id_t> new_sort; 
    vertex_id_t minLength = 20;
    vertex_id_t init_round = 5;
    
    void get_new_sort()
    {
        
        struct vocab_vertex
        {
            int cn;
            vertex_id_t id;
            vocab_vertex(){};
            vocab_vertex(uint _cn, vertex_id_t _id) : cn(_cn), id(_id){};
        };

        vector<vocab_vertex>v_vocab(this->vertex_cn.size());
        for(int i =0;i<v_vocab.size();i++)
        {
            v_vocab[i].cn = vertex_cn[i];
            v_vocab[i].id = i;
        } 
        for(int p=0;p<this->partition_num;p++)
        {
            sort(v_vocab.begin()+this->vertex_partition_begin[p],v_vocab.begin()+this->vertex_partition_end[p],[](vocab_vertex a,vocab_vertex b){
                return a.cn > b.cn;
            });
        }
        this->new_sort.resize(this->v_num);
        for(int v =0;v< this-> v_num;v++)
        {
            new_sort[v] = v_vocab[v].id;
        }
        cout<<"generate new sort in runtime\n";

    }


#ifdef COLLECT_WALK_SEQUENCE
    std::vector<std::vector<Footprint> > footprints;
#endif
#ifdef COLLECT_WALKER_INIT_STATE
    std::vector<Walker<walker_data_t> > walker_init_state;
#endif

public:
    double other_time = 0.0;
    partition_id_t get_local_partition_id()
    {
        return this->local_partition_id;
    }

    WalkEngine()
    {
        // timer = new Timer();
        randgen = new StdRandNumGenerator[this->worker_num];
    }

    ~WalkEngine()
    {

        if (randgen != nullptr)
        {
            delete []randgen;
        }
        size_t glb_cross;
        MPI_Allreduce(&this->cross_num, &glb_cross, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        size_t glb_intr;
        MPI_Allreduce(&this->intr_num, &glb_intr, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        size_t glb_trace;
        MPI_Allreduce(&this->trace_num, &glb_trace, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        size_t glb_p_step;
        MPI_Allreduce(&this->p_step, &glb_p_step, 1, get_mpi_data_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        // printf("[p%u]【sum_p_step】 %zu 【sum cross】 %zu 【sum intr】 %zu 【sum_trace】 %zu cross_num %zu intr_num %zu  trace_num %zu p_step_num %zu\n",this->local_partition_id, glb_p_step, glb_cross, glb_intr, glb_trace, cross_num, intr_num, trace_num, p_step);

       
    }

    void set_concurrency(int worker_num_param)
    {
        this->set_graph_engine_concurrency(worker_num_param);
        delete []randgen;
        randgen = new StdRandNumGenerator[worker_num_param];
    }

    void set_init_round(vertex_id_t init_round){
        if(init_round!=0)
            this->init_round = init_round;
    }

    void set_minLength(vertex_id_t minLength){
        if(minLength!=0)
            this->minLength = minLength;
    }
    StdRandNumGenerator* get_thread_local_rand_gen()
    {
        return &randgen[omp_get_thread_num()];
    }
    std::function<vertex_id_t (walker_id_t)> get_equal_dist_func()
    {
        auto equal_dist_func = [&] (walker_id_t w_id)
        {
            vertex_id_t start_v = w_id % this->v_num;
            return start_v;
        };
        return equal_dist_func;
    }

    template<typename transition_config_t>
    void random_walk(WalkerConfig<edge_data_t, walker_data_t> *walker_config, transition_config_t *transition_config, WalkConfig *walk_config_param = nullptr)
    {
        WalkConfig* walk_config = walk_config_param;
        if (walk_config_param == nullptr)
        {
            walk_config = new WalkConfig();
        }
        internal_random_walk_wrap(walker_config, transition_config, walk_config);
        if (walk_config_param == nullptr)
        {
            delete walk_config;
        }
    }

private:
// init_walkers(walk_data.local_walkers, walk_data.local_walkers_bak, walker_begin, 
// walker_begin + walk_data.active_walker_num, walker_config->walker_init_dist_func, walker_config->walker_init_state_func);
    walker_id_t init_walkers(
        Message<Walker<walker_data_t> >* &local_walkers,
        Message<Walker<walker_data_t> >* &local_walkers_bak,
        walker_id_t walker_begin,
        walker_id_t walker_end,
        std::function<vertex_id_t (walker_id_t)> walker_init_dist_func,
        std::function<void (Walker<walker_data_t>&, vertex_id_t)> walker_init_state_func
    )
    {
        typedef Walker<walker_data_t> walker_t;
        typedef Message<walker_t> walker_msg_t;

        walker_id_t local_walker_num = 0;

        auto msg_producer = [&] (void) {
                #pragma omp parallel for
                for (walker_id_t w_i = walker_begin / this->partition_num * this->partition_num + this->local_partition_id; w_i < walker_end; w_i += this->partition_num)
                {
                    if (w_i < walker_begin)
                    {
                        continue;
                    }
                    vertex_id_t start_v = walker_init_dist_func(w_i);
                    Walker<walker_data_t> walker;
                    walker.id = w_i;
                    walker.step = 0;
                    walker.setData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    assert(start_v < this->v_num);
                    this->emit(start_v, walker, omp_get_thread_num());
                #ifdef COLLECT_WALK_SEQUENCE
                    footprints[omp_get_thread_num()].push_back(Footprint(walker.id, start_v, walker.step));
                #endif
                }
            };
        auto msg_consumer = [&](walker_msg_t *begin, walker_msg_t *end)
            {
                local_walker_num = end - begin;
                std::swap(local_walkers, local_walkers_bak);
            };
        this->template distributed_execute<walker_t>(
            msg_producer,
            msg_consumer,
            local_walkers_bak
        );
        if (walker_init_state_func != nullptr)
        {
            #pragma omp parallel for
            for (walker_id_t w_i = 0; w_i < local_walker_num; w_i ++)
            {
                walker_init_state_func(local_walkers[w_i].data, local_walkers[w_i].dst_vertex_id);
            }
        }
        #ifdef COLLECT_WALKER_INIT_STATE
        walker_init_state.clear();
        for (walker_id_t w_i = 0; w_i < local_walker_num; w_i ++)
        {
            walker_init_state.push_back(local_walkers[w_i].data);
        }
        #endif
        return local_walker_num;
    }

  
public:
    template<typename query_data_t, typename response_data_t>
    struct InternalWalkData
    {
        typedef Walker<walker_data_t> walker_t;
        typedef Message<walker_t> walker_msg_t;
        Timer timer;

        walker_msg_t *local_walkers;
        walker_msg_t *local_walkers_bak;
        walker_id_t local_walker_num;
        walker_id_t active_walker_num;

        bool collect_path_flag;
        PathCollector *pc;
    };
    template<typename query_data_t, typename response_data_t, typename transition_config_t>
    void internal_random_walk(WalkerConfig<edge_data_t, walker_data_t> *walker_config, transition_config_t *transition_config, WalkConfig* walk_config, int order)
    {
        typedef Walker<walker_data_t> walker_t;
        typedef Message<walker_t> walker_msg_t;

        walker_id_t walker_num = walker_config->walker_num * init_round;
        // walker_id_t walker_per_iter = walker_num * walk_config->rate;
        walker_id_t walker_per_iter = walker_num;
        if (walker_per_iter == 0) walker_per_iter = 1;
        if (walker_per_iter > walker_num) walker_per_iter = walker_num;
        size_t walker_array_size = walker_per_iter;
        walker_id_t remained_walker = walker_num;

        InternalWalkData<query_data_t, response_data_t> walk_data;

        walk_data.collect_path_flag = false;
        walk_data.pc = nullptr;
        if (walk_config->output_file_flag || walk_config->output_consumer_flag)
        {
            walk_data.collect_path_flag = true;
        }
        if (walker_config->walker_init_dist_func == nullptr)
        {
            walker_config->walker_init_dist_func = this->get_equal_dist_func();
        }
        walk_data.local_walkers = this->template alloc_array<Message<Walker<walker_data_t> > >(walker_array_size);
        walk_data.local_walkers_bak = this->template alloc_array<Message<Walker<walker_data_t> > >(walker_array_size);

        size_t max_msg_size = 0;
        if (order == 1)
        {
            max_msg_size = sizeof(walker_msg_t);
        }
        this->set_msg_buffer(walker_array_size, max_msg_size);

        int iter = 0;
        std::vector<vertex_id_t> context_map_freq(this->v_num, 0);
        std::vector<double> H;
        bool terminal_flag = false;
        while (remained_walker != 0)
        {
            if (walk_data.collect_path_flag)
            {
                walk_data.pc = new PathCollector(this->worker_num);
            }
            walker_id_t walker_begin = walker_num - remained_walker;
            walk_data.active_walker_num = std::min(remained_walker, walker_per_iter);
            remained_walker -= walk_data.active_walker_num;
            std::cout << "walk_data.active_walker_num = " << walk_data.active_walker_num << std::endl;

            walk_data.local_walker_num = init_walkers(walk_data.local_walkers, walk_data.local_walkers_bak, walker_begin, walker_begin + walk_data.active_walker_num, walker_config->walker_init_dist_func, walker_config->walker_init_state_func);

            if (walk_data.collect_path_flag)
            {
                std::cout << "walk_data.local_walker_num = " << walk_data.local_walker_num << std::endl;
#pragma omp parallel for
                for (walker_id_t w_i = 0; w_i < walk_data.local_walker_num; w_i++)
                {
                    walk_data.pc->add_footprint(Footprint(walk_data.local_walkers[w_i].data.id, walk_data.local_walkers[w_i].dst_vertex_id, 0), omp_get_thread_num());
                }
            }
            internal_walk_epoch(&walk_data, walker_config, transition_config);

            if (walk_data.collect_path_flag)
            {
                Timer timer_ap;
                auto* paths = walk_data.pc->assemble_path(walker_begin);
                this->other_time+= timer_ap.duration();

                if (walk_config->output_file_flag)
                {
                    // std::string local_output_path = walk_config->output_path_prefix + "." + std::to_string(this->local_partition_id);
                    std::string local_output_path = walk_config->output_path_prefix;
                    Timer timer_dump;
                    paths->dump(local_output_path.c_str(), iter == 0 ? "w": "a", walk_config->print_with_head_info, context_map_freq,this->local_corpus,this->vertex_cn,this->co_occor);
                    this->other_time += timer_dump.duration();
                    
                    MPI_Allreduce(context_map_freq.data(),  this->vertex_freq, this->v_num, get_mpi_data_type<vertex_id_t>(), MPI_SUM, MPI_COMM_WORLD);
                    uint64_t words_sum = 0;
                    uint64_t degree_sum = 0;
                    for(int i = 0; i < this->v_num; i++)
                    {
                        words_sum += this->vertex_freq[i];
                        assert(words_sum >= 0);
                        assert(words_sum < UINT64_MAX);
                        degree_sum += this->vertex_out_degree[i];
                        assert(degree_sum >= 0);
                        assert(degree_sum < UINT64_MAX);
                    }
                    std::cout << "words_sum = " << words_sum << " degree_sum = " << degree_sum << std::endl;
                    double h = 0.0;
                    for(int i = 0; i < this->v_num; i++)
                    {
                        if(this->vertex_out_degree[i] <= 0){
                            std::cout << "this->vertex_out_degree[ " << i << " ] = " << this->vertex_out_degree[i] << std::endl;
                        }
                        assert(this->vertex_out_degree[i] >= 0);
                        double pi = static_cast<double>(this->vertex_out_degree[i]) / degree_sum;
                        if(pi <= 0 || pi >= 1){
                            std::cout << "pi = " << pi << std::endl;
                        }
                        assert(pi > 0 && pi <1);
                        double qi =  static_cast<double>(this->vertex_freq[i]) / words_sum;
                        if(qi == 0){std::cout << "this->vertex_freq[ " << i << " ] = " << this->vertex_freq[i] <<" degree: "<<this->vertex_out_degree[i]<< std::endl;}
                        if(qi <= 0 || qi >= 1){
                            std::cout << "qi = " << qi << std::endl;
                        }
                        assert(qi > 0 && qi < 1);
                        h += pi * (log(pi) - log(qi));
                    } 
                    
                    H.push_back(h);
                    double delta_H = H.size() == 1 ? H[H.size() - 1] : H[H.size() - 1] - H[H.size() - 2];
                    std::cout << "abs(delta_H) = " << abs(delta_H) << std::endl;
                    assert(abs(delta_H) >= 0);
                    if(this->local_partition_id == 0)
                    {
                        std::cout << "Delat RE：" << abs(delta_H) << std::endl;
                    }
                    iter = iter == 0 ? init_round + 1 : iter + 1;
                    if(abs(delta_H) <= delta_R)
                    {
                        terminal_flag = true;
                        // remained_walker = 0;
                    }else
                    {
                        if(this->local_partition_id == 0)
                        {
                            std::cout << "***********************************************Round " << iter <<" ************************************************" << std::endl;
                        }
                    }
                }
                if (walk_config->output_consumer_flag)
                {
                    walk_config->output_consumer_func(paths);
                } else
                {
                    delete paths;
                }
                if(terminal_flag == false)
                {
                    this->dealloc_array(walk_data.local_walkers, walker_array_size);
                    this->dealloc_array(walk_data.local_walkers_bak, walker_array_size);
                    walker_array_size = walker_config->walker_num;
                    walk_data.local_walkers = this->template alloc_array<Message<Walker<walker_data_t> > >(walker_array_size);
                    walk_data.local_walkers_bak = this->template alloc_array<Message<Walker<walker_data_t> > >(walker_array_size);
                    // this->free_msg_buffer();
                    // this->set_msg_buffer(walker_array_size, max_msg_size);
                    walker_num = walker_config->walker_num;
                    remained_walker = walker_num;
                }
                delete walk_data.pc;
            }
        }

        this->dealloc_array(walk_data.local_walkers, walker_array_size);
        this->dealloc_array(walk_data.local_walkers_bak, walker_array_size);
    }

    void internal_random_walk_wrap (WalkerConfig<edge_data_t, walker_data_t> *walker_config, TransitionConfig<edge_data_t, walker_data_t> *transition_config, WalkConfig *walk_config)
    {
        internal_random_walk<EmptyData, EmptyData> (walker_config, transition_config, walk_config, 1);
    }

    template<typename query_data_t, typename response_data_t>
    void internal_walk_epoch(
        InternalWalkData<query_data_t, response_data_t> *walk_data,
        WalkerConfig<edge_data_t, walker_data_t> *walker_config,
        TransitionConfig<edge_data_t, walker_data_t> *transition_config
    )
    {
        typedef Walker<walker_data_t> walker_t;
        typedef Message<walker_t> walker_msg_t;

        auto extension_comp_func = transition_config->extension_comp_func;
        auto static_comp_func = transition_config->static_comp_func;
        auto dynamic_comp_func = transition_config->dynamic_comp_func;
   

        auto walker_update_state_func = walker_config->walker_update_state_func;

        auto* local_walkers = walk_data->local_walkers;
        auto* local_walkers_bak = walk_data->local_walkers_bak;
        auto local_walker_num = walk_data->local_walker_num;

        auto output_flag = walk_data->collect_path_flag;
        auto* pc = walk_data->pc;

        auto active_walker_num = walk_data->active_walker_num;    
        int super_step = 0;

        std::vector<std::unordered_map<vertex_id_t, int>> walker_to_path(active_walker_num, std::unordered_map<vertex_id_t, int>());
        
        while (active_walker_num != 0)
        {
            #ifndef UNIT_TEST
            if (this->local_partition_id == 0)
            {
                printf("step(%d), active(%u), time(%.3lf)\n", super_step++, active_walker_num, walk_data->timer.duration());
            }
            #endif
            bool use_parallel = (active_walker_num >= OMP_PARALLEL_THRESHOLD);
            // use_parallel = false;
            auto msg_producer = [&] (void) {
                walker_id_t progress = 0;
                walker_id_t data_amount = local_walker_num;
                auto *data_begin = local_walkers;
                const walker_id_t work_step_length = PARALLEL_CHUNK_SIZE;

                #pragma omp parallel if (use_parallel)
                {
                    int worker_id = omp_get_thread_num();
                    StdRandNumGenerator* gen = get_thread_local_rand_gen();
                    vertex_id_t next_workload;
                    while((next_workload =  __sync_fetch_and_add(&progress, work_step_length)) < data_amount)
                    {
                        walker_msg_t *begin = data_begin + next_workload;
                        walker_msg_t *end = data_begin + std::min(next_workload + work_step_length, data_amount);
                        for (walker_msg_t *p = begin; p != end; p++)
                        {
                            walker_t walker = p->data;
                            vertex_id_t current_v = p->dst_vertex_id;
                            Correlation cal_corr;
                            
                            double fi = static_cast<double>(walker_to_path[walker.id][current_v]++);
                            assert(fi >= 0);
                            if(walker.N == 0)
                            {
                                cal_corr.SetProperty(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
                                // double Hn, double Sn, double EnH, double EnS, double EnHS, double EnHH, double EnSS, double R, double N
                                walker.setData(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
                   
                            }else{
                            
                                cal_corr.SetProperty(walker.Hn, walker.Sn, walker.EnH, walker.EnS, walker.EnHS, walker.EnHH, walker.EnSS, walker.N, walker.R);

                                cal_corr.iterate(fi, 1.0, worker_id);
                                walker.setData(cal_corr.Hn(), cal_corr.Sn(), cal_corr.EnH(), cal_corr.EnS(), cal_corr.EnHS(), cal_corr.EnHH(), cal_corr.EnSS(), cal_corr.R(), cal_corr.N());
                                walker.step++;
                                __sync_fetch_and_add(&p_step, 1);
                                // this->vertex_cn[current_v]++;
                                pc->add_footprint(Footprint(walker.id, current_v, walker.step), worker_id);

                                if(walker.step > minLength)
                                {
                                    if(pow(walker.R, 2) < R || walker.R < 0)
                                    {
                                        continue;
                                    }
                                }

                                if(walker.traceFlag == true)
                                {
                                     __sync_fetch_and_add(&cross_num, 1);
                                    walker.traceFlag = false;
                                    this->emit(walker.last_vertex, walker, worker_id);
                                    continue;
                                }
                            }
                    
                            while (true)
                            {
                                vertex_id_t degree = this->vertex_out_degree[current_v];
                                if (degree == 0)
                                {
                                    std::cout << "this->vertex_out_degree[" << current_v << "] = 0" << std::endl; 
                                    break;
                                } 
                                AdjList<edge_data_t> *adj = this->csr->adj_lists + current_v;
                                AdjUnit<edge_data_t> *candidate = nullptr;
                                AdjUnit<edge_data_t> *ac_edge = nullptr;
                                
                                bool isTrace = false; 

                                candidate = adj->begin + gen->gen(degree);
                                if(degree == 2 && adj->begin->neighbour == current_v){
                                    break;
                                }

                                if(walker.step == 0 && candidate->neighbour == current_v){
                                    continue;
                                }
                                // assert(this->commonNeighbors.find({current_v, candidate->neighbour}) != this->commonNeighbors.end());      
                                // int common_neighbors = this->commonNeighbors[{current_v, candidate->neighbour}];
                                // U u1 = {current_v, candidate->neighbour};
                                // U u2 = {candidate->neighbour, current_v};
                                // if(this->vertex_partition_id[candidate->neighbour] == this->local_partition_id)
                                // {
                                //     assert(this->commonNeighbors[u1.key] == this->commonNeighbors[u2.key]);
                                // }
                                // com_neighbour_t src_dst_com_nei = current_v;
                                // src_dst_com_nei = (src_dst_com_nei << 32) | candidate->neighbour;
                                // int common_neighbors = this->commonNeighbors[u.key];
                               
                                // int common_neighbors = this->commonNeighbors[u1.key];
                                int common_neighbors = candidate->data;
                                // if(current_v==37398 && candidate->neighbour == 101267){
                                //     // 37398 101267 29
                                //     std::cout << "com = " << common_neighbors << std::endl;
                                //     assert(common_neighbors==29);
                                // }
                                
                                // int common_neighbors = this->commonNeighbors[src_dst_com_nei];
                                double p = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

                                int src_degree = this->vertex_out_degree[current_v];
                                int dst_degree = this->vertex_out_degree[candidate->neighbour];

                                double p_accept = (1 / (static_cast<double>(src_degree - common_neighbors)) * (static_cast<double>(max(src_degree, dst_degree))) / min(src_degree, dst_degree));
                                double p_norm = tanh(p_accept);//biaozhunhua

                                ac_edge = candidate;

                                if(p_norm <= p)
                                {
                                    isTrace = true;
                                    __sync_fetch_and_add(&trace_num, 1);
                                }
                                vertex_id_t dst = ac_edge->neighbour;
                                // std::cout << "[ msg_producer] " << " dst " << dst << std::endl;
                                // walker.step++;
                                // pc->add_footprint(Footprint(walker.id, dst, walker.step), worker_id);
                                if(!this->is_local_vertex(dst))
                                {
                                    // std::cout << "[msg_producer] " << " emit walker to dst point " << dst << std::endl;
                                    // assert(walker.Hn > 0);
                                    if(isTrace)
                                    {
                                        walker.last_vertex = current_v;
                                        walker.traceFlag = true;
                                    }
                                    this->emit(dst, walker, worker_id); 
                                    __sync_fetch_and_add(&cross_num, 1);
                                    break;
                                }
                            
                                walker.step++;
                                __sync_fetch_and_add(&p_step, 1);
                                // this->vertex_cn[dst]++;
                                pc->add_footprint(Footprint(walker.id, dst, walker.step), worker_id);
                                double fi = static_cast<double>(walker_to_path[walker.id][dst]++);
                                assert(fi >= 0);
                                __sync_fetch_and_add(&intr_num, 1);
                                cal_corr.iterate(fi, 1.0, worker_id);

                                walker.setData(cal_corr.Hn(), cal_corr.Sn(), cal_corr.EnH(), cal_corr.EnS(), cal_corr.EnHS(), cal_corr.EnHH(), cal_corr.EnSS(), cal_corr.R(), cal_corr.N());
                                assert(walker.Hn > 0);
                                assert(walker.Sn > 1.0);
                                assert(walker.EnH > 0);
                                assert(walker.EnS > 0);
                                assert(walker.EnHS > 0);
                                assert(walker.EnSS > 1);
                                assert(walker.EnHH > 0);

                                if(walker.step > minLength)
                                {
                                    if(pow(walker.R, 2) < R || walker.R < 0)
                                    {
                                        break;
                                    }
                                }
                                if(isTrace == true)
                                {
                                    __sync_fetch_and_add(&p_step, 1);
                                    walker.step++;
                                    // this->vertex_cn[current_v]++;
                                    pc->add_footprint(Footprint(walker.id, current_v, walker.step), worker_id);
                                    __sync_fetch_and_add(&intr_num, 1);

                                    double fi = static_cast<double>(walker_to_path[walker.id][current_v]++);
                                
                                    // std::cout << "[ msg_producer ] " << " fi (must greater than 1 ) " << fi << std::endl;
                                    assert(fi > 0);
                                    

                                    cal_corr.iterate(fi, 1.0, worker_id);

                                    walker.setData(cal_corr.Hn(), cal_corr.Sn(), cal_corr.EnH(), cal_corr.EnS(), cal_corr.EnHS(), cal_corr.EnHH(), cal_corr.EnSS(), cal_corr.R(), cal_corr.N());
                                    assert(walker.Hn > 0);
                                    assert(walker.Sn > 1.0);
                                    assert(walker.EnH > 0);
                                    assert(walker.EnS > 0);
                                    assert(walker.EnHS > 0);
                                    assert(walker.EnSS > 1);
                                    assert(walker.EnHH > 0);

                                    if(walker.step > minLength){
                                        if(pow(walker.R, 2) < R || walker.R < 0)
                                        {
                                            break;
                                        }
                                    }
                                }
                                if(isTrace == false)
                                {
                                    current_v = dst;
                                }else{
                                    current_v = current_v;
                                }
                            }
                        }
                        this->notify_progress(begin - data_begin, end - data_begin, data_amount, active_walker_num >= PHASED_EXECTION_THRESHOLD * this->partition_num);
                    }
                }
                local_walker_num = 0;
            };
            auto msg_consumer = [&](Message<Walker<walker_data_t> > *begin, Message<Walker<walker_data_t> > *end)
            {
                local_walker_num = end - begin;
                std::swap(local_walkers, local_walkers_bak);
            };
            active_walker_num = this->template distributed_execute<walker_t>(
                msg_producer,
                msg_consumer,
                local_walkers_bak,
                active_walker_num >= PHASED_EXECTION_THRESHOLD * this->partition_num
            );
        }
    }
};
