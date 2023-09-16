# GPT metagenome

Modelling metagenomes by transformers / GPT.

The main aim of this repo is testing the feasability of the idea that we can model the transition probabilities of edges on a [de Brujin Graph(dBG)](https://en.wikipedia.org/wiki/De_Bruijn_graph) using a transformer-based model, such as GPT2. 


A de Bruijn graph is a directed graph used in bioinformatics and graph theory to represent sequence overlaps, often for DNA sequences. Each node in the graph corresponds to a fixed-length sequence (\(k\)-mer), and edges represent overlaps between these sequences. Thus, predicting the next edge is akin to predicting the next neucleotide in the DNA, that is similar to a typical NLP next token predicition task. 

