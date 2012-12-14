/*
 * Lda.hpp
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

#ifndef LDA_H
#define LDA_H

#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include "DataSet.hpp"

/**
 * Latent Dirichlet Allocation
 *
 * @see David M. Blei, Andrew Y. Ng, and Michael I. Jordan. Latent Dirichlet allocation. Journal of Machine Learning Research, 3:993-1022, January 2003.
 * @see Thomas P. Minka. Estimating a Dirichlet distribution. (2000; revised 2003, 2009, 2012)
 * @see Thomas L. Griffiths, and Mark Steyvers. Finding scientific topics.
 */
class Lda {
    DataSet dataset;
    DataSet testset;
    const int K;
    std::vector<double> alpha_z;
    double beta;

    std::vector<std::vector<int>> n_m_z;
    std::vector<std::vector<int>> n_z_t;
    std::vector<int> n_z;
    std::vector<std::vector<int>> z_m_n;

    std::vector<std::vector<double>> phi_z_t;
    std::vector<std::vector<double>> theta_m_z;

    bool asymmetry;

    // random number generator
    std::mt19937 gen;

    void init();
    void sampling_z(const int m, const int n);
    void update_alpha();

public:
    Lda(const unsigned int _K, const double _alpha, const double _beta, unsigned int _seed,
            const char *train, const char *test, const char *vocab, bool asymmetry);
    virtual ~Lda() = default;
    void inference();
    double perplexity();
    void learn(const unsigned int iteration);
    void dump();
};

#endif
