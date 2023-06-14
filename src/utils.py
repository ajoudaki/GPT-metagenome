# utils.py
import torch
from collections import defaultdict, Counter
from typing import List
import random


def compute_char_probabilities(sequences: List[str]) -> None:
    counts = defaultdict(int)
    total_count = 0
    for seq in sequences:
        cd = Counter(seq)
        for c, count in cd.items():
            counts[c] += count
            total_count += count
    pcounts = {char: c / total_count for char, c in counts.items()}
    print(pcounts)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
