# What is LDA?
See Wikipedia, [Latent Dirichlet allocation](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation).

# What is HDP?
See Wikipedia, [Hierarchical Dirichlet process](http://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process).

# System Requirements
* Compiler that supports C++11
* Boost C++ Libraries  
\*NOTICE\* Boost C++ Libraries shoud be built with the C++11 compiler.

# Usage
See `--help`.

# Data Set
## Format
The 1st line:        the number of docs  
The 2nd line:        the number of vocabulary  
The 3rd line:        the number of words (\*NOTICE\* not NNZ, the number of nonzero counts in the bag-of-words)  
The following lines: docID wordID count

## Vocabulary
line number = wordID

## For example
[UCI Machine Learning Repository: Bag of Words Data Set](http://archive.ics.uci.edu/ml/datasets/Bag+of+Words)

# Licence
MIT License  
Copyright (c) 2012 Tsukasa ŌMOTO([@henry0312](https://twitter.com/henry0312))

# Special Thanks To
* Mr. Shuyo Nakatani([@shuyo](https://twitter.com/shuyo)) / Cybozu Labs Inc.  
I consulted his implementation, <https://github.com/shuyo/iir/tree/master/lda>.  
* Mr. Hiroki Taniura([@boiled_sugar](https://twitter.com/boiled_sugar), <https://github.com/boiled-sugar>)  
I had my Enlgish translation corrected.  
* Mr. Jan Ekström([@jeebjp](https://twitter.com/jeebjp), <https://github.com/jeeb>)  
English adviser
* Mr. Motofumi Oka([@mtfmk](https://twitter.com/mtfmk), <https://github.com/chikuzen>)  
I referred to his configure and Makefile.
