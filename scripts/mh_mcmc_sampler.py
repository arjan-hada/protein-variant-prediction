# modified from: https://github.com/ElArkk/jax-unirep/blob/master/jax_unirep/sampler.py

from collections import defaultdict
from random import choice
from typing import Callable, Dict

import numpy as np
import numpy.random as npr
from multipledispatch import dispatch
from tqdm.autonotebook import tqdm

proposal_valid_letters = "ACDEFGHIKLMNPQRSTVWY"

letters_sorted = sorted(proposal_valid_letters)

aa_dict = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
}


def is_accepted(best: float, candidate: float, temperature: float) -> bool:
    """
    Return boolean decision on whether the candidate mutant is accepted or not.

    This function checks whether we want to
    accept a new mutant proposed by our MMC sampler,
    by comparing its predicted activity
    to the current best mutants activity.

    :param best: Predicted activity of current best mutant
    :param candidate: Predicted activity of new candidate mutant
    :param temperature: Boltzmann distribution temperature.
        Controls acceptance probability.
        Low T decreases acceptance probability.
        High T increases acceptance probability.
    :returns bool: Whether or not candidate mutant was accepted
    """

    c = np.exp((candidate - best) / temperature)

    if c > 1:
        return True
    else:
        p = np.random.uniform(0, 1)
        if c >= p:
            return True
        else:
            return False


@dispatch(str)
def propose(
    sequence: str, mu: float = npr.uniform(1,2.5)
) -> str:
    """
    Given a string s, return a proposed mutated string s*, by randomly adding m ~ Poisson(μ − 1) + 1 mutations, 
    where μ is the sequence proposal mutation rate.

    The proposed mutant is generated as follows:

    - Randomly add m ~ Poisson(μ − 1) + 1 mutations, where μ is the sequence proposal mutation rate.
    - Pick a position from random unifrom distribution
    - Given that position, pick a letter from random uniform distribution

    Propose a new sequence, 

    :param sequence: The sequence to propose a new mutation on.
    :param mu: sequence proposal mutation rate. 
    Default value of mu was set to be a random draw from a uniform(1,2.5) distribution.

    :returns: A string.
    """
    if len(sequence) == 0:
        raise ValueError(
            "sequence passed into `propose` must be at least length 1."
        )


    sequence_list = list(sequence)
    
    n_mutations = npr.poisson(mu-1) + 1 # number of random mutations to a given sequence
    
    for i in range(n_mutations):
        # The location as well as the mutation is sampled from a uniform random distribution.
        position = npr.randint(0, len(sequence)) 
        sequence_list[position] = choice(list(proposal_valid_letters))
        
    return ''.join(sequence_list)


def hamming_distance(s1: str, s2: str):
    """Return hamming distance between two strings of the same length."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def init_traj_seq(
    wt_sequence: str,
    trust_radius: int = 7
) -> str:
    """
    Given a string s, return a proposed mutated string s*, by randomly adding m ~ Poisson(2) + 1 random mutations.
    This will be used as initial sequence for each trajectory.
    """
    if len(wt_sequence) == 0:
        raise ValueError(
            "sequence passed into `propose` must be at least length 1."
        )
    
    starter_sequence = propose(wt_sequence, mu=3)
    
    while hamming_distance(wt_sequence, starter_sequence) > trust_radius:
        starter_sequence = propose(wt_sequence, mu=3)
        
    return starter_sequence
    
def sample_one_chain(
    wt_sequence: str,
    starter_sequence: str,
    n_steps: int,
    scoring_func: Callable,
    is_accepted_kwargs: Dict = {},
    trust_radius: int = 7,
    propose_kwargs: Dict = {},
) -> Dict:
    """
    Return one chain of MCMC samples of new sequences.

    Given a `starter_sequence`,
    this function will sample one chain of protein sequences,
    scored using a user-provided `scoring_func`.

    Design choices made here include the following.

    Firstly, we record all sequences that were sampled,
    and not just the accepted ones.
    This behaviour differs from other MCMC samplers
    that record only the accepted values.
    We do this just in case sequences that are still "good"
    (but not better than current) are rejected.
    The effect here is that we get a cluster of sequences
    that are one-apart from newly accepted sequences.

    Secondly, we check the Hamming distance
    between the newly proposed sequences and the original.
    This corresponds to the "trust radius"
    specified in the [jax-unirep paper](https://doi.org/10.1101/2020.01.23.917682).
    If the hamming distance > trust radius,
    we reject the sequence outright.

    A dictionary containing the following key-value pairs are returned:

    - "sequences": All proposed sequences.
    - "scores": All scores from the scoring function.
    - "accept": Whether the sequence was accepted as the new 'current sequence'
        on which new sequences are proposed.

    This can be turned into a pandas DataFrame.

    ### Parameters
    
    - `wt_sequence`: The wt sequence.
    - `starter_sequence`: The starting sequence.
    - `n_steps`: Number of steps for the MC chain to walk.
    - `scoring_func`: Scoring function for a new sequence.
        It should only accept a string `sequence`.
    - `is_accepted_kwargs`: Dictionary of kwargs to pass into
        `is_accepted` function.
        See `is_accepted` docstring for more details.
    - `trust_radius`: Maximum allowed number of mutations away from
        starter sequence.
    - `propose_kwargs`: Dictionary of kwargs to pass into
        `propose` function.
        See `propose` docstring for more details.
    - `verbose`: Whether or not to print iteration number
        and associated sequence + score. Defaults to False

    ### Returns

    A dictionary with `sequences`, `accept` and `score` as keys.
    """
    current_sequence = starter_sequence
    current_score = scoring_func(sequence=starter_sequence)

    chain_data = defaultdict(list)
    chain_data["sequences"].append(current_sequence)
    chain_data["scores"].append(current_score)
    chain_data["accept"].append(True)

    for i in tqdm(range(n_steps)):
        new_sequence = propose(current_sequence, **propose_kwargs)
        new_score = scoring_func(sequence=new_sequence)

        default_is_accepted_kwargs = {"temperature": 0.1}
        default_is_accepted_kwargs.update(is_accepted_kwargs)
        accept = is_accepted(
            best=current_score,
            candidate=new_score,
            **default_is_accepted_kwargs,
        )

        # Check hamming distance
        if hamming_distance(wt_sequence, new_sequence) > trust_radius:
            accept = False

        # Determine acceptance
        if accept:
            current_sequence = new_sequence
            current_score = new_score

        # Record data.
        chain_data["sequences"].append(new_sequence)
        chain_data["scores"].append(new_score)
        chain_data["accept"].append(accept)
    chain_data["scores"] = np.hstack(chain_data["scores"])
    return chain_data