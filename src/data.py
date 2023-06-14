# data.py
import torch
import random
from collections import defaultdict, Counter, List, Tuple
from Bio import SeqIO
from torch.nn import functional as F


def read_fna(file_path: str, shuffle: bool = False) -> List[str]:
    sequences = []
    with open(file_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append(str(record.seq))
    if shuffle:
        random.shuffle(sequences)
    return sequences


def generate_markov_chain(n: int, sparsity: float) -> torch.Tensor:
    transition_probs = torch.rand(n, n)
    for i in range(n):
        num_outgoing_states = torch.poisson(torch.tensor([float(sparsity)])).int().item()
        num_outgoing_states = min(n, max(1, num_outgoing_states))
        _, indices = torch.topk(transition_probs[i], num_outgoing_states)
        mask = torch.zeros_like(transition_probs[i]).scatter_(0, indices, 1).to(torch.bool)
        transition_probs[i] *= mask
    transition_probs = F.normalize(transition_probs, p=1, dim=1)
    return transition_probs


def draw_seq(transition_probs: torch.Tensor, sequence_length: int) -> str:
    NUCLEOTIDES = ['A', 'C', 'G', 'T']
    current_state = 0
    chain = [current_state]
    while len(chain) < sequence_length:
        next_state = torch.multinomial(transition_probs[current_state], num_samples=1)
        current_state = next_state.item()
        chain.append(current_state)
    seq = [NUCLEOTIDES[s % len(NUCLEOTIDES)] for s in chain]
    return ''.join(seq)


def generate_HMM_dataset(sequence_length: int, N: int, sparsity: float, num_hidden_states: int = None) -> Tuple[List[str], List[str]]:
    if num_hidden_states is None:
        num_hidden_states = sequence_length
    transition_matrix = generate_markov_chain(num_hidden_states, sparsity=sparsity)
    all_seqs = [[], []]
    for _ in range(2):
        seqs = []
        while len(seqs) < N:
            seq = draw_seq(transition_matrix, sequence_length)
            if len(seq) >= sequence_length:
                seqs.append(seq)
        all_seqs[_] = seqs
    return all_seqs


def generate_phylo_dataset(sequence_length: int, mutation_rate: float, N: int) -> List[str]:
    NUCLEOTIDES = ['A', 'C', 'G', 'T']
    parent_sequence = ''.join(random.choice(NUCLEOTIDES) for _ in range(sequence_length))
    dataset = [parent_sequence]
    for _ in range(N):
        mutated_sequence = ''
        for nucleotide in parent_sequence:
            if random.random() < mutation_rate:
                mutated_sequence += random.choice(NUCLEOTIDES)
            else:
                mutated_sequence += nucleotide
        dataset.append(mutated_sequence)
        parent_sequence = mutated_sequence
    return dataset
