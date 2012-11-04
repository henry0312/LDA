/*
 * HdpLda.hpp
 *
 * Copyright (c) 2012 Tsukasa OMOTO <henry0312@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/* This file is available under an MIT license. */

#ifndef HDP_LDA_H
#define HDP_LDA_H

#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include "DataSet.hpp"
#include "BetaDistribution.hpp"

class HdpLda {
    DataSet dataset;
    DataSet testset;

    double alpha;
    const double alpha_a; // shape parameter
    const double alpha_b; // scale parameter
    double beta;
    double gamma;
    const double gamma_a; // shape parameter
    const double gamma_b; // scale parameter

    std::vector<std::vector<int>> tables;
    // size of tables in the j-th restaurant, not the number of tables.
    std::vector<int> m_j;
    std::vector<int> dishes;
    int K;  // size of dishes, not the number of topics

    std::vector<std::vector<int>> t_j_i;

    std::vector<std::vector<int>> n_j_t;
    std::vector<std::vector<std::vector<int>>> n_j_t_v;

    std::vector<int> n_k;
    std::vector<std::vector<int>> n_k_v;

    std::vector<std::vector<int>> k_j_t;

    int m;  // the number of tables that all the restaurants have
    std::vector<int> m_k;

    std::vector<std::vector<double>> phi_k_v;
    std::vector<std::vector<double>> theta_j_k;

    // random number generator
    std::random_device rd;
    std::mt19937 gen;

    void init();
    void sampling_t(const int j, const int i);
    void sampling_k(const int j, const int t);
    void remove_table(const int j, const int t);
    void remove_dish(const int k);
    int assign_new_dish();
    int add_new_table(const int j, const int k);
    int get_new_dish();
    int get_empty_table(const int j);
    void update_alpha();
    void update_gamma();

public:
    HdpLda(const double _alpha, const double _alpha_a, const double _alpha_b,
            const double _beta, const double _gamma, const double _gamma_a,
            const double _gamma_b, const char *train, const char *test, const char *vocab);
    ~HdpLda() = default;
    void inference();
    double perplexity();
    void learn(const int iteration);
    void dump();
    int count_topics();
};

#endif
