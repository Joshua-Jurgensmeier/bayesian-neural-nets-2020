import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from random import random

# Random walk Metropolis Markov Chain
# Takes a bnn in and generates a markov chain with equilibrium equal to the
# posterior distribution of network parameters.
class Metropolis:
    def __init__(self, state, data, target_mask, posterior_dist, device):
        self.data = data
        self.target_mask = target_mask
        self.crnt_state = state
        self.posterior_dist = posterior_dist
        self.crnt_posterior = posterior_dist(data, target_mask, state)
        self.device = device
        self.scale = torch.tensor([0.00002], device=self.device)
        
        self.transitions = 0

    # Generate a new parameter state, sampling from a normal distribution centered
    # at (has the mean of) our current state.
    def sample_proposal(self):
        proposal = {}
        for key, param in self.crnt_state.items():
            # Expand std scalar up to size of parameter
            std = self.scale.expand_as(param.data)
            proposal[key] = nn.Parameter(torch.normal(param.data, std)).to(self.device)

        return proposal

    # Performs one markov chain step, potentially transitioning state
    def step(self):
        # Sample proposal
        prop_state = self.sample_proposal()
        # Compute posterior
        prop_posterior = self.posterior_dist(self.data, self.target_mask, prop_state)
        # Accept or reject proposal with probability equal to ratio of
        # proposal posterior to current state posterior
        ratio = prop_posterior - self.crnt_posterior
        threshold = torch.log(torch.rand(1, dtype=torch.double))
        if torch.isfinite(ratio) and threshold < ratio:
            #print("Ratio is {}-{}={} and threshold is {}, transitioning.".format(prop_posterior, self.crnt_posterior, ratio, threshold))
            self.transitions += 1
            self.crnt_state = prop_state
            self.crnt_posterior = prop_posterior
        # else:
        #     print("Ratio is {}-{}={} and threshold is {}, not transitioning.".format(prop_posterior, self.crnt_posterior, ratio, threshold))

    # Steps the chain num_steps number of times and returns a list
    # of the states reached, sampling weights from the posterior
    # If burn is true, then samples are discarded.
    def sample_chain(self, num_steps, burn=False):
        samples = []
        for i in range(num_steps):
            self.step()
            if not burn:
                samples.append(self.crnt_state)
        
        print("Generated samples. Transitioned {}/{}".format(self.transitions, num_steps))
        return samples
