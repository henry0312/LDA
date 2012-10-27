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
#include "DataSet.hpp"

class Lda {
    DataSet dataset;
    DataSet testset;
    const int K;
    double alpha;
    double beta;

    std::vector<std::vector<int>> n_m_z;
    std::vector<std::vector<int>> n_z_t;
    std::vector<int> n_z;
    std::vector<std::vector<int>> z_m_n;

    std::vector<std::vector<double>> phi_z_t;
    std::vector<std::vector<double>> theta_m_z;

    /*
     * random number generator
     */
    std::random_device rd;
    std::mt19937 gen;

    void init();
public:
    Lda(const int K, double alpha, double beta,
            const char *train, const char *test, const char *vocab);
    virtual ~Lda() = default;
    void inference();
    double perplexity();
    void learn(const int iteration);
    void dump();
};

#endif
