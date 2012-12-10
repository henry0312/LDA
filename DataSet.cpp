/*
 * DataSet.cpp
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

#include "DataSet.hpp"

/**
 * Constructor
 *
 * Load DataSet
 *
 * @param const char *dataset DataSet's filename
 */
DataSet::DataSet(const char *dataset)
    :M(0), V(0), N(0)
{
    loadDataSet(dataset);
}

/**
 * Constructor
 *
 * Load DataSet and Vocabulary
 *
 * @param const char *dataset DataSet's filename
 * @param const char *vocab Vocabulary's filename
 */
DataSet::DataSet(const char *dataset, const char *vocab)
    :M(0), V(0), N(0)
{
    loadDataSet(dataset);
    loadVocabulary(vocab);
}

/**
 * Load a file and Initialize variables
 *
 * @param const char *filename open *filename
 */
void DataSet::loadDataSet(const char *filename) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Can't open the file: " << filename << std::endl;
        exit(1);
    }

    // the 1st line : the number of docs
    fin >> M;
    docs.resize(M);
    n_m.resize(M, 0);

    // the 2nd line : the number of vocabulary
    fin >> V;

    // the 3rd line : the number of words
    fin >> N;

    // the following lines : docID wordID count
    int m, v, cnt;
    while ( fin >> m >> v >> cnt ) {
        for (int i = 0; i < cnt; ++i) {
            docs[m-1].push_back(v);
            ++n_m[m-1];
        }
    }

    fin.close();
}

/**
 * Loac Vocabulary
 *
 * @param const char *filename open *filename
 */
void DataSet::loadVocabulary(const char *filename) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Can't open the file: " << filename << std::endl;
        exit(1);
    }

    std::string buff;
    while ( fin >> buff ) {
        vocab.push_back(buff);
    }

    fin.close();
}
