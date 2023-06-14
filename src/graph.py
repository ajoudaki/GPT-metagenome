# graph.py
from collections import defaultdict, Set, List
import re


def construct_debruijn_graph(dataset: List[str], k: int) -> defaultdict[str, Set[str]]:
    graph = defaultdict(set)
    for sequence in dataset:
        for i in range(len(sequence) - k):
            graph[sequence[i:i + k]].add(sequence[i + 1:i + k + 1])
    return graph


def extract_substrings(sequences: List[str], sequence_length: int, stride: int, substrings_per_seq: int) -> List[str]:
    substrings = []
    for sequence in sequences:
        for i in range(0, len(sequence) - sequence_length + 1, stride):
            if i // stride > substrings_per_seq:
                break
            seq = sequence[i:i + sequence_length]
            if bool(re.match("^[ACGT]+$", seq)):
                substrings.append(seq)
    return substrings
