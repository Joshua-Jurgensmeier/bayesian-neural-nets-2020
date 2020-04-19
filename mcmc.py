import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from random import random
import copy

# Random walk Metropolis Markov Chain
# Takes a bnn in and generates a markov chain with equilibrium equal to the
# posterior distribution of network parameters.
class Metropolis:
    def __init__(self, model):
        print("Initializing the Markov Chain...")
        self.model = model
        self.proposal_dist = dist.Normal(0, 1)
        self.crnt_state = self.model.state_dict()
        self.crnt_posterior = self.model.posterior(self.crnt_state)
        print("Markov Chain initialized")

    def sample_proposal(self):
        proposal = copy.deepcopy(self.model.state_dict())
        for param in proposal.values():
            param.data = self.proposal_dist.sample(param.data.size())
        return proposal

    # Performs one markov chain step, potentially transitioning state
    def step(self):
        # Sample proposal
        prop_state = self.sample_proposal()
        # Compute posterior
        prop_posterior = self.model.posterior(prop_state)
        # Accept or reject proposal with probability equal to ratio of
        # proposal posterior to current state posterior
        ratio = prop_posterior - self.crnt_posterior
        if torch.isfinite(ratio) and torch.log(torch.rand(1)) < ratio:
            self.crnt_state = prop_state
            self.crnt_posterior = prop_posterior

    # Steps the chain num_steps number of times and returns a list
    # of the states reached, sampling weights from the posterior
    # If burn is true, then samples are discarded.
    def sample_chain(self, num_steps, burn=False):
        samples = []
        for i in range(num_steps):
            print("Generating sample {}".format(i))
            self.step()
            if not burn:
                samples.append(self.crnt_state)
        return samples
