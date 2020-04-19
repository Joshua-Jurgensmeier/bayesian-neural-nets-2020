import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import mcmc

# A Bayesian Neural Network model with a zero mean guassian weight prior and logsoftmax likelihood.
# Hidden layers use ReLU and output is categorical output with logsoftmax
class BayesNet(nn.Module):
    def __init__(self, data_loader, D_h, D_h2, D_out, prior=None):
        super().__init__()

        self.data_loader = data_loader

        # Get size of input in the most esoteric and
        # elaborate way possible
        sample = next(iter(data_loader))
        D_in = sample[0].view(-1).size()[0]

        # Set prior
        if prior is not None:
            self.prior = prior
        else:
            # Who knows if this is a good one
            self.prior = dist.Normal(0, 1)

        # Add linear sub-modules. These store our parameters.
        self.lin1 = nn.Linear(D_in, D_h)
        self.lin2 = nn.Linear(D_h, D_h2)
        self.lin3 = nn.Linear(D_h2, D_out)

        # Initalize the Markov Chain
        self.mc = mcmc.Metropolis(self)
        self.samples = []

        # No gradients for us, ladies and gentlemen.
        for param in self.parameters():
            param.requires_grad = False

    # Passes input through network, returning categorization tensor
    def forward(self, x):
        y = F.relu(self.lin1(x.view(-1)))
        y = F.relu(self.lin2(y))
        y = F.log_softmax(self.lin3(y), dim=0)
        return y

    # Compute P(Wb|data) = P(data|Wb)P(Wb)
    # That is, take in a weight matrix and dataset, then
    # return the posterior probability of the weight matrix,
    # which is (proportional to) the likelihood times the prior.
    # All probabilities are represented as log probabilities.
    def posterior(self, param_dict=None):
        # Update parameters if passed in, otherwise use the ones we
        # already have.
        if param_dict is not None:
            self.load_state_dict(param_dict, strict=True)

        # Compute prior of parameters
        prior_lp = 0
        for param in self.parameters():
            prior_lp += torch.sum(self.prior.log_prob(param.data)).item()

        # Compute likelihood(parameters|data_loader). The likelihood
        # is the model's probability for the correct output 
        like_lp = sum(self.forward(x)[y] for x, y in self.data_loader)

        return prior_lp + like_lp

    # Initializes Markov Chain, training network on data. After calling
    # this, predict may be called to estimate the posterior predictive
    # distribution.
    def train(self, burn_in=500, samples=280):
        self.mc.sample_chain(burn_in, burn=True)
        self.samples = self.mc.sample_chain(samples)

    # Uses Markov Chain to estimate the posterior predictive distribution and
    # returns the best guess, as well as uncertainty, if Markov Chain has
    # been initialized.
    def predictive(self, x):
        out = torch.zeros(self.lin3.bias.size())
        
        # Average network outputs with sampled parameters
        for sample in self.samples:
            self.load_state_dict(sample, strict=True)
            out += self.forward(x)
        
        out /= float(len(self.samples))

        return out.argmax(dim=0)
