/*
 * HdpLdaMain.cpp
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

#include <iostream>
#include <string>
#include <random>
#include <boost/program_options.hpp>
#include "HdpLda.hpp"

int main(int argc, char const* argv[])
{
    using namespace std;
    using namespace boost::program_options;

    // Set options
    options_description opt("Options");
    opt.add_options()
        ("help,h",                                              "show help")
        ("alpha,a",     value<double>(),                        "alpha")
        ("alpha_shape", value<double>()->default_value(1),      "shape parameter, alpha is drawn from Gamma(alpha_shape, slpha_scale)")
        ("alpha_scale", value<double>()->default_value(1),      "scale parameter, alpha is drawn from Gamma(alpha_shape, alpha_scale)")
        ("beta,b",      value<double>()->default_value(0.5),    "beta")
        ("gamma,g",     value<double>(),                        "gamma")
        ("gamma_shape", value<double>()->default_value(1),      "shape parameter, gamma is drawn from Gamma(gamma_shape, gamma_scale)")
        ("gamma_scale", value<double>()->default_value(1),      "scale parameter, gamma is drawn from Gamma(gamma_shape, gamma_scale)")
        ("seed,s",      value<unsigned int>(),                  "seed value to use in the initialization of the internal state of std::mt19937. if not set, std::random_device is used for the initialization.")
        ("iteration,i", value<int>()->default_value(10),        "the number of times of inference")
        ("train",       value<string>(),                        "Training set")
        ("test",        value<string>(),                        "Test set")
        ("vocab",       value<string>(),                        "Vocabulary");

    // Parse the arguments and Store the result in vm.
    variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);

    if ( vm.count("help") || !vm.count("train") || !vm.count("test") || !vm.count("vocab") ) {
        cout << opt << endl;
        return 1;
    }

    /*
     * Set the parameters
     */
    double alpha_shape  = vm["alpha_shape"].as<double>();
    double alpha_scale  = vm["alpha_scale"].as<double>();
    double beta         = vm["beta"].as<double>();
    double gamma_shape  = vm["gamma_shape"].as<double>();
    double gamma_scale  = vm["gamma_scale"].as<double>();
    const int i         = vm["iteration"].as<int>();
    string train        = vm["train"].as<string>();
    string test         = vm["test"].as<string>();
    string vocab        = vm["vocab"].as<string>();
    // alpha
    double alpha = 0.0;
    if (vm.count("alpha")) {
        alpha = vm["alpha"].as<double>();
    } else {
        alpha = alpha_shape * alpha_scale; // mean
    }
    // gamma
    double gamma = 0.0;
    if (vm.count("gamma")) {
        gamma = vm["gamma"].as<double>();
    } else {
        gamma = gamma_shape * gamma_scale; // mean
    }
    // seed
    unsigned int seed = 0;
    if (vm.count("seed")) {
        seed = vm["seed"].as<unsigned int>();
    } else {
        std::random_device rd;
        seed = rd();
    }

    // HDP-LDA
    HdpLda hdplda(alpha, alpha_shape, alpha_scale, beta, gamma,
            gamma_shape, gamma_scale, seed, train.c_str(), test.c_str(), vocab.c_str());
    hdplda.learn(i);

    return 0;
}
