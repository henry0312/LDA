/*
 * Main.cpp
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
#include <boost/program_options.hpp>
#include "Lda.hpp"

int main(int argc, char const* argv[])
{
    using namespace std;
    using namespace boost::program_options;

    // Set options
    options_description opt("Options");
    opt.add_options()
        ("help,h",                                              "show help")
        ("topic,K",     value<int>()->default_value(30),        "the number of topics")
        ("alpha,a",     value<double>()->default_value(0.5),    "alpha")
        ("beta,b",      value<double>()->default_value(0.5),    "beta")
        ("iteration,i", value<int>()->default_value(10),        "the number of times of inference")
        ("train",       value<string>(),                        "Training set")
        ("test",        value<string>(),                        "Test set");

    // Parse the arguments and Store the result in vm.
    variables_map vm;
    store(parse_command_line(argc, argv, opt), vm);
    notify(vm);

    if ( vm.count("help") || !vm.count("train") || !vm.count("test") ) {
        cout << opt << endl;
        return 1;
    }

    // Set the parameters
    const int K     = vm["topic"].as<int>();
    double alpha    = vm["alpha"].as<double>();
    double beta     = vm["beta"].as<double>();
    const int i     = vm["iteration"].as<int>();
    string train    = vm["train"].as<string>();
    string test     = vm["test"].as<string>();

    // LDA
    Lda lda(K, alpha, beta, train.c_str(), test.c_str());
    lda.learn(i);

    return 0;
}
