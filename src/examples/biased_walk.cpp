#include "walk.hpp"
#include "option_helper.hpp"

int main(int argc, char** argv)
{
    MPI_Instance mpi_instance(&argc, &argv);

    TruncatedRandomWalkOptionHelper opt;
    opt.parse(argc, argv);

    WalkEngine<real_t, EmptyData> graph;
    graph.load_graph(opt.v_num, opt.graph_path.c_str(), opt.make_undirected);
    WalkConfig walk_conf;
    if (!opt.output_path.empty())
    {
        walk_conf.set_output_file(opt.output_path.c_str());
    }
    if (opt.set_rate)
    {
        walk_conf.set_walk_rate(opt.rate);
    }
    WalkerConfig<real_t, EmptyData> walker_conf(opt.walker_num);
    auto extension_comp = [&] (Walker<EmptyData>& walker, vertex_id_t current_v)
    {
        return walker.step >= opt.walk_length ? 0.0 : 1.0; /*walk opt.walk_length steps then terminate*/
    };
    auto static_comp = [&] (vertex_id_t v, AdjUnit<real_t> *edge)
    {
        return edge->data; /*edge->data is a real number denoting edge weight*/
    };
    TransitionConfig<real_t, EmptyData> tr_conf(extension_comp, static_comp);
    graph.random_walk(&walker_conf, &tr_conf, &walk_conf);
    return 0;
}
