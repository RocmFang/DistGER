

#pragma once

#include <assert.h>

#include <iostream>
#include <thread>

#include <args/args.hxx>

#include "type.hpp"

class OptionHelper
{
protected:
    //The order of class member variable initiation
    //depends on the order of member variable declaration in the class
    //so parser should be placed before any other args' flags
    args::ArgumentParser parser;
private:
    args::HelpFlag help;
public:
    OptionHelper() :
        parser("", ""),
        help(parser, "help", "Display this help menu", {'h', "help"})
    {}

    virtual void parse(int argc, char **argv)
    {
        try
        {
            parser.ParseCLI(argc, argv);
        }
        catch (args::Help)
        {
            std::cout << parser;
            exit(0);
        }
        catch (args::ParseError e)
        {
            // std::cerr << e.what() << std::endl;
            // std::cerr << parser;
            // exit(1);
        }
        catch (args::ValidationError e)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << parser;
            exit(1);
        }
    }
};
// class EmbeddingOptionHelper : public OptionHelper
// {
// private:
//     args::ValueFlag<int> size_flag;
//     args::ValueFlag<std::string> train_flag;
//     args::ValueFlag<std::string> save_vocab_flag;
//     args::ValueFlag<std::string> read_vocab_flag;
//     args::ValueFlag<int> debug_flag;
//     args::ValueFlag<float> alpha_flag;
//     args::ValueFlag<std::string> embout_flag;
//     args::ValueFlag<int> window_flag;
//     args::ValueFlag<int> sample_flag;
//     args::ValueFlag<int> negative_flag;
//     args::ValueFlag<int> threads_flag;
//     args::ValueFlag<int> iter_flag;
//     args::ValueFlag<int> min_count_flag;
//     args::ValueFlag<int> batch_size_flag;
//     args::ValueFlag<int> disk_flag;

// public:
//     int size;
//     std::string train;
//     std::string save_vocab;
//     std::string read_vocab;
//     int debug;
//     float alpha;
//     std::string embout;
//     int window;
//     int sample;
//     int negative;
//     int threads;
//     int iter;
//     int min_count;
//     int batch_size;
//     int disk;
//     EmbeddingOptionHelper():
//         size(parser,"size","",)
//     {}
// };

class GraphOptionHelper : public OptionHelper
{
private:
    args::ValueFlag<vertex_id_t> v_num_flag;
    args::ValueFlag<std::string> graph_path_flag;
    args::ValueFlag<std::string> partition_path_flag;
    args::ValueFlag<vertex_id_t> min_length_flag;
    args::ValueFlag<vertex_id_t> init_round_flag;
    args::Flag make_undirected_flag;

public:
    vertex_id_t v_num;
    std::string graph_path;
    std::string partition_path;
    bool make_undirected;
    vertex_id_t min_length;
    vertex_id_t init_round;
    GraphOptionHelper():
        v_num_flag(parser, "vertex", "vertex number", {'v'}),
        graph_path_flag(parser, "graph", "graph data path", {'g'}),
        partition_path_flag(parser,"partition","graph vertices partition path",{'p'}),
        min_length_flag(parser, "min_length","min length per walker",{"min_L"}),
        init_round_flag(parser, "init_round", "init round to begin", {"min_R"}),

        make_undirected_flag(parser, "make-undirected", "load graph and treat each edge as undirected edge", {"make-undirected"})
    {}
    virtual void parse(int argc, char **argv)
    {
        OptionHelper::parse(argc, argv);

        assert(v_num_flag);
        v_num = args::get(v_num_flag);

        assert(min_length_flag);
        min_length = args::get(min_length_flag);

        assert(init_round_flag);
        init_round = args::get(init_round_flag);

        assert(graph_path_flag);
        graph_path = args::get(graph_path_flag);

        partition_path = args::get(partition_path_flag);

        make_undirected = make_undirected_flag;
    }
};

class RandomWalkOptionHelper : public GraphOptionHelper
{
private:
    args::ValueFlag<walker_id_t> walker_num_flag;
    args::ValueFlag<std::string> output_path_flag;
    args::ValueFlag<double> rate_flag;
public:
    walker_id_t walker_num;
    std::string output_path;
    double rate;
    bool set_rate;
    RandomWalkOptionHelper() :
        walker_num_flag(parser, "walker", "walker number", {'w'}),
        output_path_flag(parser, "output", "[optional] the output path. Omit this option for pure random walk performance testing without output.", {'o'}),
        rate_flag(parser, "rate", "Set this option will break random walk into multiple iterations to save memory. Each iteration has rate% walkers.", {'r'})
    {}

    virtual void parse(int argc, char** argv)
    {
        GraphOptionHelper::parse(argc, argv);

        assert(walker_num_flag);
        walker_num = args::get(walker_num_flag);

        if (output_path_flag)
        {
            output_path = args::get(output_path_flag);
        }

        if (rate_flag)
        {
            set_rate = true;
            rate = args::get(rate_flag);
        } else
        {
            set_rate = false;
        }
    }
};

class SRandomWalkOptionHelper : public RandomWalkOptionHelper
{
private:
    args::ValueFlag<std::string> static_comp_flag;
public:
    std::string static_comp;
    SRandomWalkOptionHelper() :
        static_comp_flag(parser, "static_comp", "[weighted | unweighted] a weighted graph usually indicates a non-trivial static component.", {'s'})
    {}
    virtual void parse(int argc, char** argv)
    {
        RandomWalkOptionHelper::parse(argc, argv);
        assert(static_comp_flag);
        static_comp = args::get(static_comp_flag);
        assert(static_comp.compare("weighted") == 0 || static_comp.compare("unweighted") == 0);
    }
};

class TruncatedRandomWalkOptionHelper : public RandomWalkOptionHelper
{
private:
    args::ValueFlag<step_t> walk_length_flag;
public:
    step_t walk_length;
    TruncatedRandomWalkOptionHelper() :
        walk_length_flag(parser, "length", "walk length", {'l'})
    {}

    virtual void parse(int argc, char **argv)
    {
        RandomWalkOptionHelper::parse(argc, argv);

        assert(walk_length_flag);
        walk_length = args::get(walk_length_flag);
    }
};

class STruncatedRandomWalkOptionHelper : public TruncatedRandomWalkOptionHelper
{
private:
    args::ValueFlag<std::string> static_comp_flag;
public:
    std::string static_comp;
    STruncatedRandomWalkOptionHelper() :
        static_comp_flag(parser, "static_comp", "[weighted | unweighted] a weighted graph usually indicates a non-trivial static component.", {'s'})
    {}
    virtual void parse(int argc, char** argv)
    {
        TruncatedRandomWalkOptionHelper::parse(argc, argv);
        assert(static_comp_flag);
        static_comp = args::get(static_comp_flag);
        assert(static_comp.compare("weighted") == 0 || static_comp.compare("unweighted") == 0);
    }
};
