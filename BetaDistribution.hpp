/*
 * BetaDistribution.hpp
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

#ifndef BETA_DISTRIBUTION_H
#define BETA_DISTRIBUTION_H

#include <random>

template <class RealType = double>
class beta_distribution
{
    std::gamma_distribution<> g_alpha;
    std::gamma_distribution<> g_beta;
public:
    explicit beta_distribution(const RealType alpha, const RealType beta);
    ~beta_distribution() = default;
    template <class Generator> double operator()(Generator& gen);
};

/**
 * Constructor
 *
 * @param const doulbe alpha is one of shape parameters, Beta(alpha, beta).
 * @param const double beta is the other shape parameters, Beta(alpha, beta)
 * @see http://en.wikipedia.org/wiki/Beta_distribution
 */
template <class RealType>
beta_distribution<RealType>::beta_distribution(const RealType alpha, const RealType beta)
    :g_alpha(alpha, 1), g_beta(beta, 1)
{
}

/**
 * Generates the next random number in the distribution
 *
 * @param Generator& gen an uniform random number generator object
 * @see http://en.wikipedia.org/wiki/Gamma_distribution
 * @see http://en.wikipedia.org/wiki/Gamma_distribution#Others
 */
template <class RealType>
template <class Generator>
double beta_distribution<RealType>::operator()(Generator& gen) {
    double x = g_alpha(gen);
    double y = g_beta(gen);
    return x / (x + y);
}

#endif

