/*
 * HdpLda.cpp
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

#include "HdpLda.hpp"

/**
 * Constructor
 *
 * @param const double _alpha alpha
 * @param const double _alpha_a shape parameter
 * @param const double _alpha_b scale parameter
 * @param const double _beta beta
 * @param const double _gamma gamma
 * @param const double _gamma_a shape parameter
 * @param const double _gamma_b scale parameter
 * @param const char *train Training set
 * @param const char *test Test set
 * @param const char *vocab Vocabulary
 */
HdpLda::HdpLda(const double _alpha, const double _alpha_a, const double _alpha_b,
        const double _beta, const double _gamma, const double _gamma_a,
        const double _gamma_b, const char *train, const char *test, const char *vocab)
    :dataset(train, vocab), testset(test), alpha(_alpha), alpha_a(_alpha_a), alpha_b(_alpha_b),
    beta(_beta), gamma(_gamma), gamma_a(_gamma_a), gamma_b(_gamma_b), K(0), m(0), gen(rd())
{
    init();
}

/**
 * Initialization
 */
void HdpLda::init() {
    // tables
    tables.resize(dataset.M);

    // m_j
    m_j.resize(dataset.M, 0);

    // t_j_i
    t_j_i.resize(dataset.M);
    for (int j = 0; j < dataset.M; j++) {
        // -1 means not assigned
        t_j_i[j].resize(dataset.n_m[j], -1);
    }

    // n_j_t
    n_j_t.resize(dataset.M);
    for (auto& n_t : n_j_t) {
        n_t.resize(1);
    }

    // n_j_t_v
    n_j_t_v.resize(dataset.M);
    for (auto& n_t_v : n_j_t_v) {
        n_t_v.resize(1);
        for (auto& n_v : n_t_v) {
            n_v.resize(dataset.V, 0);
        }
    }

    // n_k
    n_k.resize(1);

    // n_k_v
    n_k_v.resize(1);
    for (auto& n_v : n_k_v) {
        n_v.resize(dataset.V, 0);
    }

    // m_k
    m_k.resize(1, 0);

    // k_jt
    k_j_t.resize(dataset.M);
    for (auto& k_t : k_j_t) {
        k_t.resize(1);
    }

    // theta
    theta_j_k.resize(dataset.M);
}

/**
 * Inference
 */
void HdpLda::inference() {
    /*
     * sampling t_ji
     */
    for (int j = 0; j < dataset.M; j++) {
        for (int i = 0; i < dataset.n_m[j]; i++) {
            sampling_t(j, i);
        }
    }

    /*
     * sampling k_jt
     */
    for (int j = 0; j < dataset.M; j++) {
        for (int t = 0; t < m_j[j]; t++) {
            if (tables[j][t] == 1) {
                sampling_k(j, t);
            }
        }
    }

    /*
     * Update hyperparameters
     */
    update_gamma();
    update_alpha();
}

/**
 * Sampling t_ji
 *
 * @param const int j the j-th doc(restaurant)
 * @param const int i the i-th word(guest) in the j-th doc(restaurant)
 */
void HdpLda::sampling_t(const int j, const int i) {
    const int old_t = t_j_i[j][i];
    const int old_k = k_j_t[j][old_t];
    const int v = dataset.docs[j][i] - 1;

    /*
     * Decrease counters
     */
    if (old_t >= 0) {
        n_k[old_k]--;
        n_k_v[old_k][v]--;
        n_j_t[j][old_t]--;
        n_j_t_v[j][old_t][v]--;

        if (n_j_t[j][old_t] == 0) {
            remove_table(j, old_t);
        }
    }

    /*
     * Sampling
     */
    // f_k
    std::vector<double> f_k(K);
    for (int k = 0; k < K; k++) {
        f_k[k] = (beta + n_k_v[k][v]) / (dataset.V * beta + n_k[k]);
    }

    // p_x
    double p_x = 0.0;
    for (int k = 0; k < K; k++) {
            p_x += m_k[k] * f_k[k];
    }
    p_x += gamma / dataset.V;
    p_x /= gamma + m;

    // p_t
    std::vector<double> p_t(m_j[j] + 1);
    for (int t = 0; t < m_j[j]; t++) {
        p_t[t] = n_j_t[j][t] * f_k[ k_j_t[j][t] ];
    }
    p_t[m_j[j]] = alpha * p_x;

    // sampling
    std::discrete_distribution<> dis_p_t(begin(p_t), end(p_t));
    int new_t = dis_p_t(gen);

    // new_t == t^new
    if (new_t  == m_j[j]) {
        /*
         * Sampling k_jt^new
         */
        // p_k_jt^new
        std::vector<double> p_k(K+1);
        for (int k = 0; k < K; k++) {
            p_k[k] = m_k[k] * f_k[k];
        }
        p_k[K] = gamma / dataset.V;

        // sampling
        std::discrete_distribution<> dis_p_k(begin(p_k), end(p_k));
        int new_k = dis_p_k(gen);

        // new_k == k^new
        if (new_k == K) {
            new_k = assign_new_dish();
        }

        new_t = add_new_table(j, new_k);
    }

    /*
     * Update and Increase counters
     */
    const int new_k = k_j_t[j][new_t];
    t_j_i[j][i] = new_t;
    n_j_t[j][new_t]++;
    n_k[new_k]++;
    n_k_v[new_k][v]++;
    n_j_t_v[j][new_t][v]++;
}

/**
 * Remove an empty table
 *
 * @param const int j the j-th doc(restaurant)
 * @param const int t the t-th table in the j-th doc(restaurant)
 */
void HdpLda::remove_table(const int j, const int t) {
    const int k = k_j_t[j][t];

    // Update and Decrease counters
    tables[j][t] = 0;
    m--;
    m_k[k]--;
    if (m_k[k] == 0) {
        remove_dish(k);
    }
}

/**
 * Assign a new dish(topic) to a table
 *
 * @return a new dish(topic)
 */
int HdpLda::assign_new_dish() {
    const int new_k = get_new_dish();

    // new dish
    if (new_k == K) {
        dishes.resize(new_k + 1);
        K = dishes.size();
        m_k.resize(new_k + 1);
        n_k.resize(new_k + 1);
        n_k_v.resize(new_k + 1);
        n_k_v[new_k].resize(dataset.V, 0);
    }

    // Update
    dishes[new_k] = 1;

    return new_k;
}

/**
 * Add a new table to a restaurant
 *
 * @param const int j the j-th doc(restaurant)
 * @param const int k a dish(topic) to be set on the table
 * @return a new table(new_t)
 */
int HdpLda::add_new_table(const int j, const int k) {
    const int new_t = get_empty_table(j);

    // new table
    if (new_t == m_j[j]) {
        tables[j].resize(new_t + 1);
        m_j[j] = tables[j].size();
        k_j_t[j].resize(new_t + 1);
        n_j_t[j].resize(new_t + 1);
        n_j_t_v[j].resize(new_t + 1);
        n_j_t_v[j][new_t].resize(dataset.V, 0);
    }

    // Update and Increase counters
    tables[j][new_t] = 1;
    k_j_t[j][new_t] = k;
    m++;
    m_k[k]++;

    return new_t;
}

/**
 * Get a dish that haven't been set yet or a new dish
 *
 * @return a new dish(topic)
 */
int HdpLda::get_new_dish() {
    for (int k = 0; k < K; k++) {
        if (dishes[k] == 0) {
            return k;
        }
    }
    return dishes.size();
}

/**
 * Get an empty table
 *
 * @return an empty table
 */
int HdpLda::get_empty_table(const int j) {
    for (int t = 0; t < m_j[j]; t++) {
        if (tables[j][t] == 0) {
            return t;
        }
    }
    return tables[j].size();
}

/**
 * Sampling k_jt
 *
 * @param const int j the j-th doc(restaurant)
 * @param const int t the t-th table in the j-th doc(restaurant)
 */
void HdpLda::sampling_k(const int j, const int t) {
    const int old_k = k_j_t[j][t];
    const int n_jt = n_j_t[j][t];

    /*
     * Decrease counters
     */
    n_k[old_k] -= n_jt;
    for (int v = 0; v < dataset.V; v++) {
        n_k_v[old_k][v] -= n_j_t_v[j][t][v];
    }
    m_k[old_k]--;
    if (m_k[old_k] == 0) {
        remove_dish(old_k);
    }

    /*
     * Sampling
     */
    // f_k
    double numer, denom;
    std::priority_queue<double> queue_f_k;
    std::vector<double> f_k(K+1);
    for (int k = 0; k < K; k++) {
        if (m_k[k] == 0) {
            f_k[k] = 1;
            continue;
        }
        numer = denom = 0.0;
        for (int n = 0; n < n_j_t[j][t]; n++) {
            denom += std::log(dataset.V * beta + n_k[k] + n);
        }
        for (int v = 0; v < dataset.V; v++) {
            for (int n = 0; n < n_j_t_v[j][t][v]; n++) {
                numer += std::log(beta + n_k_v[k][v] + n);
            }
        }
        f_k[k] = numer - denom;
        queue_f_k.push(f_k[k]);
    }

    // f_k^new
    numer = denom = 0.0;
    for (int n = 0; n < n_j_t[j][t]; n++) {
        denom += std::log(dataset.V * beta + n);
    }
    for (int v = 0; v < dataset.V; v++) {
        for (int n = 0; n < n_j_t_v[j][t][v]; n++) {
            numer += std::log(beta + n);
        }
    }
    f_k[K] = numer - denom;
    queue_f_k.push(f_k[K]);

    // normalizing
    double max_f_k = queue_f_k.top();
    for (int k = 0; k < K; k++) {
        if (m_k[k] != 0) {
            f_k[k] = std::exp(f_k[k] - max_f_k);
        }
    }
    f_k[K] = std::exp(f_k[K] - max_f_k);

    // p_k
    std::vector<double> p_k(K+1);
    for (int k = 0; k < K; k++) {
        p_k[k] = m_k[k] * f_k[k];
    }
    p_k[K] = gamma * f_k[K];

    // sampling
    std::discrete_distribution<> dis(begin(p_k), end(p_k));
    int new_k = dis(gen);

    // new_k == k^new
    if (new_k == K) {
        new_k = assign_new_dish();
    }

    /*
     * Update and Increase counters
     */
    k_j_t[j][t] = new_k;
    m_k[new_k]++;
    n_k[new_k] += n_jt;
    for (int v = 0; v < dataset.V; v++) {
        n_k_v[new_k][v] += n_j_t_v[j][t][v];
    }
}

/**
 * Remove a dish
 *
 * @param const int k a dish(topic) to be removed.
 */
void HdpLda::remove_dish(const int k) {
    dishes[k] = 0;
}

/**
 * Compute Perplexity
 */
double HdpLda::perplexity() {
    /*
     * phi
     */
    phi_k_v.resize(K+1);
    for (int k = 0; k < K; k++) {
        if (dishes[k] == 1) {
            phi_k_v[k].resize(dataset.V);
            for (int v = 0; v < dataset.V; v++) {
                phi_k_v[k][v] = (beta + n_k_v[k][v]) / (dataset.V * beta + n_k[k]);
            }
        }
    }
    // k^new
    phi_k_v[K].resize(dataset.V, 1.0/dataset.V);

    /*
     * theta
     */
    for (int j = 0; j < dataset.M; j++) {
        theta_j_k[j].resize(K+1);
        // calc n_jk
        for (int t = 0; t < m_j[j]; t++) {
            if (tables[j][t] == 1) {
                int k = k_j_t[j][t];
                theta_j_k[j][k] += n_j_t[j][t];
            }
        }
        for (int k = 0; k < K; k++) {
            if (dishes[k] == 1) {
                theta_j_k[j][k] += alpha * m_k[k] / (gamma + m);
                theta_j_k[j][k] /= dataset.n_m[j] + alpha;
            }
        }
        // k^new
        theta_j_k[j][K] = alpha * gamma / (gamma + m);
        theta_j_k[j][K] /= dataset.n_m[j] + alpha;
    }

    /*
     * Perplexity
     */
    double log_per = 0.0;
    for (int j = 0; j < testset.M; j++) {
        for (int i = 0; i < testset.n_m[j]; i++) {
            int v = testset.docs[j][i] - 1;
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                if (dishes[k] == 1) {
                    sum += theta_j_k[j][k] * phi_k_v[k][v];
                }
            }
            // k^new
            sum += theta_j_k[j][K] * phi_k_v[K][v];

            log_per -= log(sum);
        }
    }

    return exp( log_per / testset.N );
}

/**
 * Learning
 *
 * Make inferences specified number of times and Calculate perplexity with each cycle
 *
 * @param const int iteration the number of times of inference
 */
void HdpLda::learn(const int iteration) {
    using namespace std;

    cout.precision(3);
    cout.setf(ios::fixed);

    // Start time
    auto start = std::chrono::system_clock::now();

    // Inference
    std::cout << "iter\talpha\tgamma\ttopics\tperplexity\n";
    for (int i = 1; i <= iteration; i++) {
        cout << i << "\t" << alpha << "\t" << gamma << "\t";
        inference();
        cout << count_topics() << "\t" << perplexity() << endl;
    }

    // End time
    auto end = std::chrono::system_clock::now();

    // Elapsed time
    auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(end - start).count();
    int s = ms * 0.001; ms -= s * 1000;
    int m = s / 60; s %= 60;
    int h = m / 60; m %= 60;
    cout << "Elapsed time: " << h << "h " << m << "m " << s << "." << ms << "s\n" << endl;

    // Dump
    dump();
}

/**
 * Print topic-word distribution
 */
void HdpLda::dump() {
    // Insert topic-word distribution to vector
    std::vector<std::vector<std::pair<int, double>>> topic_word;
    topic_word.resize(K);
    for (int k = 0; k < K; k++) {
        if (dishes[k] == 1) {
            topic_word[k].resize(dataset.V);
            for (int v = 0; v < dataset.V; v++) {
                topic_word[k][v] = std::make_pair(v, phi_k_v[k][v]);
            }
        }
    }

    // Descending sort
    for (int k = 0; k < K; k++) {
        if (dishes[k] == 1) {
            std::sort( begin(topic_word[k]), end(topic_word[k]),
                        [](const std::pair<int, double> &a, const std::pair<int, double> &b)
                        -> bool { return a.second > b.second; } );
        }
    }

    // Print
    for (int k = 0; k < K; k++) {
        if (dishes[k] == 1) {
            printf("Topic: %d (%d words)\n", k, n_k[k]);
            for (int i = 0; i < (n_k[k] > 10 ? 10 : n_k[k]); i++) {
                auto v = topic_word[k][i].first;
                auto phi = topic_word[k][i].second;
                printf("%s: %f (%d)\n", dataset.vocab[v].c_str(), phi, n_k_v[k][v]);
            }
            std::cout << std::endl;
        }
    }
}

/**
 * Get the number of topics
 *
 * @return the number of topics
 */
int HdpLda::count_topics() {
    int topics = 0;
    for (auto k : dishes) {
        if (k == 1) {
            topics++;
        }
    }
    return topics;
}

/**
 * Sampling new alpha
 */
void HdpLda::update_alpha() {
    double sum_log_w, sum_s;
    for (int step = 0; step < 20; step++) {
        sum_log_w = sum_s = 0.0;
        for (int j = 0; j < dataset.M; j++) {
            beta_distribution<> beta_dist(alpha + 1, dataset.n_m[j]);
            sum_log_w += std::log( beta_dist(gen) );

            std::bernoulli_distribution bernoulli_dist( (double)dataset.n_m[j] / (dataset.n_m[j] + alpha) );
            sum_s += bernoulli_dist(gen);
        }
        std::gamma_distribution<> gamma_dist( alpha_a + m - sum_s, 1.0 / (alpha_b - sum_log_w) );
        alpha = gamma_dist(gen);
    }
}

/**
 * Sampling new gamma
 */
void HdpLda::update_gamma() {
    beta_distribution<> beta_dist(gamma + 1, m);
    const double eta = beta_dist(gen);

    const int k = count_topics();
    const double pi = (gamma_a + k - 1) / ( (gamma_a + k - 1) + m * (gamma_b - std::log(eta)) );

    std::gamma_distribution<> gamma_dist1( gamma_a + k, 1.0 / (gamma_b - std::log(eta)) );
    std::gamma_distribution<> gamma_dist2(  gamma_a + k - 1, 1.0 / (gamma_b - std::log(eta)) );
    gamma = pi * gamma_dist1(gen) + (1 - pi) * gamma_dist2(gen);
}

