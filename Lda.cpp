/*
 * Lda.cpp
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

#include "Lda.hpp"

/**
 * Constructor
 *
 * @param const int K the number of Topics
 * @param int alpha alpha
 * @param int beta beta
 * @param const char *train Training set
 * @param const char *test Test set
 */
Lda::Lda(const int K, double alpha, double beta, const char *train, const char *test)
    :dataset(train), testset(test), K(K), alpha(alpha), beta(beta), gen(rd())
{
    init();
}

/**
 * Initialization
 */
void Lda::init() {
    // n_mz
    n_m_z.resize(dataset.M);
    for (int m = 0; m < dataset.M; m++) {
        n_m_z[m].resize(K, 0);
    }

    // n_zt
    n_z_t.resize(K);
    for (int z = 0; z < K; z++) {
        n_z_t[z].resize(dataset.V, 0);
    }

    // n_z
    n_z.resize(K, 0);

    /*
     * Topics
     */
    int z;
    z_m_n.resize(dataset.M);
    std::uniform_int_distribution<> dis(0, K-1);
    for (int m = 0; m < dataset.M; m++) {
        z_m_n[m].resize(dataset.n_m[m]);
        for (int n = 0; n < dataset.n_m[m]; n++) {
            z = dis(gen);
            z_m_n[m][n] = z;
            n_m_z[m][z]++;
            n_z_t[z][dataset.docs[m][n] - 1]++;
            n_z[z]++;
        }
    }

    // phi
    phi_z_t.resize(K);
    for (int z = 0; z < K; z++) {
        phi_z_t[z].resize(dataset.V);
    }

    // theta
    theta_m_z.resize(dataset.M);
    for (int m = 0; m < dataset.M; m++) {
        theta_m_z[m].resize(K);
    }

}

/**
 * Inference
 *
 * perform Gibbs sampling once
 */
void Lda::inference() {
    std::vector<double> p_z(K);
    for (int m = 0; m < dataset.M; m++) {
        auto z_n = z_m_n[m];
        for (int n = 0; n < dataset.n_m[m]; n++) {
            // word
            auto t = dataset.docs[m][n];
            // old topic
            auto old_z = z_n[n];

            /*
             * Delete old topic
             */
            n_m_z[m][old_z]--;
            n_z_t[old_z][t - 1]--;
            n_z[old_z]--;

            /*
             * Gibbs sampling
             */
            for (int z = 0; z < K; z++) {
                p_z[z] = (alpha + n_m_z[m][z]) * (beta + n_z_t[z][t - 1]) / (n_z[z] + dataset.V * beta);
            }
            std::discrete_distribution<> dis(begin(p_z), end(p_z));
            auto new_z = dis(gen);

            /*
             * Update topic
             */
            z_m_n[m][n] = new_z;
            n_m_z[m][new_z]++;
            n_z_t[new_z][t - 1]++;
            n_z[new_z]++;
        }
    }
}

/**
 * compute Perplexity
 */
double Lda::perplexity() {
    /*
     * phi
     */
    for (int z = 0; z < K; z++) {
        for (int t = 0; t < dataset.V; t++) {
            phi_z_t[z][t] = (beta + n_z_t[z][t]) / (n_z[z] + dataset.V * beta);
        }
    }

    /*
     * theta
     */
    for (int m = 0; m < dataset.M; m++) {
        for (int z = 0; z < K; z++) {
            theta_m_z[m][z] = (alpha + n_m_z[m][z]) / (dataset.n_m[m] + K * alpha);
        }
    }

    /*
     * Perplexity
     */
    double log_per = 0.0;
    for (int m = 0; m < testset.M; m++) {
        for (int n = 0; n < testset.n_m[m]; n++) {
            double sum = 0.0;
            for (int z = 0; z < K; z++) {
                sum += theta_m_z[m][z] * phi_z_t[z][testset.docs[m][n] - 1];
            }
            log_per -= log(sum);
        }
    }
    return exp( log_per / testset.N );
}

/**
 * Learning
 *
 * Perform Gibbs sampling specified number of times and Calculate perplexity with each cycle
 *
 * @param int iteration the number of times of inference
 */
void Lda::learn(const int iteration) {
    using namespace std;

    // Start time
    auto start = std::chrono::system_clock::now();

    // Inference
    for (int i = 0; i < iteration; i++) {
        cout << i << "," << perplexity() << endl;
        inference();
    }
    cout << iteration << "," << perplexity() << endl;

    // End time
    auto end = std::chrono::system_clock::now();

    // Elapsed time
    auto elapsed = std::chrono::duration_cast< std::chrono::milliseconds >(end - start);
    auto s = elapsed.count() * 0.001;
    int m = s / 60;
    int h = m / 60;
    cout << "Elapsed: " << h << "h " << m << "m " << s << "s" << endl;
}
