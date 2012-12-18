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
 * @param const unsigned int _K the number of Topics
 * @param const double _alpha hyperparameter, alpha
 * @param const double _beta hyperparameter, beta
 * @param const unsigned int _seed seed value
 * @param const char *train Training set
 * @param const char *test Test set
 * @param const char *vocab Vocabulary
 * @param bool _asymmetry If true, use Asymmetry Dirichlet distribution
 */
Lda::Lda(const unsigned int _K, const double _alpha, const double _beta, const unsigned int _seed,
        const char *train, const char *test, const char *vocab, bool _asymmetry=false)
    :dataset(train, vocab), testset(test), K(_K), alpha_z(_K, _alpha),
    beta(_beta), asymmetry(_asymmetry), gen(_seed)
{
    init();
}

/**
 * Initialization
 */
void Lda::init() {
    // n_mz
    n_m_z.resize(dataset.M);
    for (auto& n_z : n_m_z) {
        n_z.resize(K, 0);
    }

    // n_zt
    n_z_t.resize(K);
    for (auto& n_t : n_z_t) {
        n_t.resize(dataset.V, 0);
    }

    // n_z
    n_z.resize(K, 0);

    /*
     * Topics
     */
    z_m_n.resize(dataset.M);
    std::uniform_int_distribution<> dis(0, K-1);
    for (int m = 0; m < dataset.M; ++m) {
        z_m_n[m].resize(dataset.n_m[m]);
        for (int n = 0; n < dataset.n_m[m]; ++n) {
            auto z = dis(gen);
            z_m_n[m][n] = z;
            ++n_m_z[m][z];
            ++n_z_t[z][dataset.docs[m][n] - 1];
            ++n_z[z];
        }
    }

    // phi
    phi_z_t.resize(K);
    for (auto& phi_t : phi_z_t) {
        phi_t.resize(dataset.V);
    }

    // theta
    theta_m_z.resize(dataset.M);
    for (auto& theta_z : theta_m_z) {
        theta_z.resize(K);
    }
}

/**
 * Inference
 */
void Lda::inference() {
    /*
     * Sampling z_mn
     */
    for (int m = 0; m < dataset.M; ++m) {
        for (int n = 0; n < dataset.n_m[m]; ++n) {
            sampling_z(m, n);
        }
    }
}

/**
 * Sampling z_mn
 *
 * Perform Gibbs sampling once
 *
 * @param const int m the mth doc
 * @param const int n the nth word
 */
void Lda::sampling_z(const int m, const int n) {
    // word
    const int t = dataset.docs[m][n];
    // old topic
    const int old_z = z_m_n[m][n];

    /*
     * Delete old topic
     */
    --n_m_z[m][old_z];
    --n_z_t[old_z][t - 1];
    --n_z[old_z];

    /*
     * Gibbs sampling
     */
    std::vector<double> p_z(K);
    for (int z = 0; z < K; ++z) {
        p_z[z] = (alpha_z[z] + n_m_z[m][z]) * (beta + n_z_t[z][t - 1]) / (n_z[z] + dataset.V * beta);
    }
    std::discrete_distribution<> dis(begin(p_z), end(p_z));
    int new_z = dis(gen);

    /*
     * Update topic
     */
    z_m_n[m][n] = new_z;
    ++n_m_z[m][new_z];
    ++n_z_t[new_z][t - 1];
    ++n_z[new_z];
}

/**
 * Compute Perplexity
 */
double Lda::perplexity() {
    /*
     * phi
     */
    for (int z = 0; z < K; ++z) {
        for (int t = 0; t < dataset.V; ++t) {
            phi_z_t[z][t] = (beta + n_z_t[z][t]) / (n_z[z] + dataset.V * beta);
        }
    }

    /*
     * theta
     */
    for (int m = 0; m < dataset.M; ++m) {
        for (int z = 0; z < K; ++z) {
            theta_m_z[m][z] = (alpha_z[z] + n_m_z[m][z]) / (dataset.n_m[m] + K * alpha_z[z]);
        }
    }

    /*
     * Perplexity
     */
    double log_per = 0.0;
    for (int m = 0; m < testset.M; ++m) {
        for (int n = 0; n < testset.n_m[m]; ++n) {
            const int t = testset.docs[m][n] - 1;
            double sum = 0.0;
            for (int z = 0; z < K; ++z) {
                sum += theta_m_z[m][z] * phi_z_t[z][t];
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
 * @param const unsigned int iteration the number of times of inference
 * @param const unsigned int burn_in burn-in period
 */
void Lda::learn(const unsigned int iteration, const unsigned int burn_in) {
    using namespace std;
    cout.setf(ios::fixed);

    /*
     * Show Initial parameters
     */
    cout << "K = " << K << endl;
    if (asymmetry) {
        cout << setprecision(6) << "alpha_z = " << alpha_z[0] << endl;
    } else {
        cout << setprecision(6) << "alpha = " << alpha_z[0] << endl;
    }
    cout << setprecision(6) << "beta = " << beta << endl;

    // Start time
    auto start = std::chrono::system_clock::now();

    // Inference
    cout.precision(3);
    cout << "iter\tperplexity\n";
    for (unsigned int i = 0; i < iteration; ++i) {
        cout << i << "\t" << perplexity() << endl;

        /*
         * Update hyperparameters
         */
        if (asymmetry) {
            if (i >= burn_in) {
                update_alpha();
            }
        }

        inference();
    }
    cout << iteration << "\t" << perplexity() << endl;

    // End time
    auto end = std::chrono::system_clock::now();

    // Elapsed time
    auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(end - start).count();
    int s = ms * 0.001; ms -= s * 1000;
    int m = s / 60; s %= 60;
    int h = m / 60; m %= 60;
    cout << "Elapsed time: " << h << "h " << m << "m " << s << "." << ms << "s\n" << endl;

    /*
     * Dump
     */
    // topic-word distribution
    dump();
    // alpha_z
    if (asymmetry) {
        cout.precision(6);
        for (int z = 0; z < K; ++z) {
            cout << "alpha_z[" << z << "] = " << alpha_z[z] << endl;
        }
    }
}

/**
 * Dump
 *
 * Print topic-word distribution
 */
void Lda::dump() {
    // Insert topic-word distribution to vector
    std::vector<std::vector<std::pair<int, double>>> topic_word;
    topic_word.resize(K);
    for (int z = 0; z < K; ++z) {
        topic_word[z].resize(dataset.V);
        for (int t = 0; t < dataset.V; ++t) {
            topic_word[z][t] = std::make_pair(t, phi_z_t[z][t]);
        }
    }

    // Descending sort
    for (int z = 0; z < K; ++z) {
        std::sort( begin(topic_word[z]), end(topic_word[z]),
                    [](const std::pair<int, double> &a, const std::pair<int, double> &b)
                    -> bool { return a.second > b.second; } );
    }

    // Print
    for (int z = 0; z < K; ++z) {
        printf("Topic: %d (%d words)\n", z, n_z[z]);
        for (int i = 0; i < (n_z[z] > 10 ? 10 : n_z[z]); ++i) {
            auto t = topic_word[z][i].first;
            auto phi = topic_word[z][i].second;
            printf("%s: %f (%d)\n", dataset.vocab[t].c_str(), phi, n_z_t[z][t]);
        }
        std::cout << std::endl;
    }
}

/**
 * Sampling new alpha
 */
void Lda::update_alpha() {
    using namespace boost::math;

    double sum_alpha = 0.0;
    for (auto alpha : alpha_z) {
        sum_alpha += alpha;
    }

    double numer, denom;
    for (int z = 0; z < K; ++z) {
        numer = denom = 0.0;
        for (int m = 0; m < dataset.M; ++m) {
            numer += digamma(n_m_z[m][z] + alpha_z[z]) - digamma(alpha_z[z]);
            denom += digamma(dataset.n_m[m] + sum_alpha) - digamma(sum_alpha);
        }
        alpha_z[z] = alpha_z[z] * numer / denom;
    }
}
