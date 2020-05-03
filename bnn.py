import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import mcmc

# The plain ol' feed-forward neural network at the core of it all
# Hidden layers use ReLU and output is categorical output with logsoftmax
class SimpleNN(nn.Module):
    def __init__(self, D_in, D_h, D_h2, D_out):
        super(SimpleNN, self).__init__()

        # Add linear sub-modules. These store our parameters.
        self.lin1 = nn.Linear(D_in, D_h)
        self.lin2 = nn.Linear(D_h, D_h2)
        self.lin3 = nn.Linear(D_h2, D_out)
    
    # Passes input through network, which is equivalent to the 
    # log_likelihood(parameters|data) 
    def forward(self, data):
        y = F.relu(self.lin1(data.view(-1, 784)))
        y = F.relu(self.lin2(y))
        y = F.log_softmax(self.lin3(y), dim=1)
        return y
        

# A Bayesian Neural Network model with a zero mean guassian weight prior and logsoftmax likelihood.
class MCMCBayesNet():
    def __init__(self, D_h, D_h2, D_out, dataset, device, prior_dist=None):
        self.dataset = dataset
        self.device = device
        
        # Get size of input in the most esoteric and
        # elaborate way possible
        sample = next(iter(dataset))
        D_in = sample[0].view(-1).size()[0]

        # The network model
        self.model = SimpleNN(D_in, D_h, D_h2, D_out).to(device)

        # Set prior. Should be a log_prob method.
        if prior_dist is not None:
            self.prior_dist = prior_dist
        else:
            # My prior for this prior is very low.
            # Should look into a better one.
            # Standard deviation is a random number
            self.prior_dist = dist.Normal(0, 17).log_prob
        
        self.chain_initialized = False

    # Return log prior of parameters w.r.t. prior_dist
    def prior(self):
        prior_lp = 0
        for param in self.model.parameters():
            prior_lp += torch.sum(self.prior_dist(param.data))
        return prior_lp

    # Do what Pytorch does best, optimization.
    # This will find the parameters with the maximum a posteriori probability
    # (MAP). This is a good starting point for our Markov Chain
    # Note, this doesn't use the posterior method, because getting the batching to
    # align correctly was too hard.
    def maximize_posterior(self, epochs, batch_size, learn_rate):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), learn_rate)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, pin_memory=True)

        # Training loop
        for epoch in range(1, epochs+1):
            print("Doing epoch {}".format(epoch))
            for data, target in dataloader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target) - self.prior()
                loss.backward()
                optimizer.step()
        
    # Compute and print some statistics on how good the current parameters are,
    # in a frequentist sense. This helps us to know if we are good to start the 
    # Markov Chain from the current state.
    def test_network(self, test_set, batches):
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=batches, pin_memory=True)
        total_loss = 0
        correct = 0
        with torch.no_grad():
            # Testing loop
            for data, target in dataloader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item() - self.prior()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()   

        avg_loss = total_loss / len(dataloader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))
        self.model.train()

    # Make a mask to select the target indices out of the output
    # of a batch. Used in the posterior function.
    def make_target_mask(self, batches, target):
        out = torch.zeros((batches, self.model.lin3.bias.size()[0]), dtype=torch.bool, device=self.device, requires_grad=False)
        for i in range(batches):
            out[i][target[i].item()] = True
        return out

    # Return posterior (actually just proportional to).
    # Compute P(parameters|data) = P(data|parameters)P(parameters)
    # Data may be a batch.
    def posterior(self, data, target_mask, param_dict=None):
        if param_dict is not None:
            self.model.load_state_dict(param_dict)

        prior_lp = self.prior()
        # Likelihood is the network output for the target category,
        # so compute all outputs (batch) and select out the target ones with
        # target_mask.
        like_lps = self.model(data)
        like_lp = like_lps.masked_select(target_mask).sum().item()

        return like_lp + prior_lp

    # Initializes Markov Chain and generates samples from the parameter
    # posterior distribution by iterating the chain. After calling
    # this, predictive may be called to estimate the posterior predictive
    # distribution.
    def init_predictive(self, samples=100):
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), pin_memory=True)
            data, target = next(iter(dataloader))
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            target_mask = self.make_target_mask(len(self.dataset), target)

            # Initalize the Markov Chain
            self.mc = mcmc.Metropolis(self.model.state_dict(), data, target_mask, self.posterior, self.device)
            # Sample from the posterior dist for parameters
            self.param_samples = self.mc.sample_chain(samples)

            self.chain_initialized = True

    # Uses Markov Chain to estimate the posterior predictive distribution and
    # returns the best guess (mean), as well as uncertainty (variance), if Markov Chain has
    # been initialized.
    def predictive(self, data):
        with torch.no_grad():
            if not self.chain_initialized:
                print("Need to initialize Markov Chain before using predictive dist")
                print("Calling init_predictive...")
                self.init_predictive()

            # Pre-allocated tensors to copy results into
            out  = torch.zeros((len(self.param_samples) * data.size()[0], self.model.lin3.bias.size()[0]), device=self.device)

            # Average network outputs with sampled parameters
            for i, sample in enumerate(self.param_samples):
                self.model.load_state_dict(sample, strict=True)
                
                sample_out = self.model(data)

                # Fancy index magic to copy results all together into out
                begin = i * data.size()[0]
                end = (i * data.size()[0]) + data.size()[0]
                out[begin:end,:] = sample_out
            
            # Compute the mean mu and and coefficient of variance std/mu,
            # where std is standard deviation
            mean = out.mean(dim=0, keepdim=True)
            # print(mean)
            cfcnt_of_var = out.std(dim=0, keepdim=True)
            # print(cfcnt_of_var)
            assert not (cfcnt_of_var != cfcnt_of_var).any()
            
            return (mean, cfcnt_of_var)

    # Test the posterior on test_set, computing some statistics to see how well we can guess,
    # and what our uncertainty is.
    # Mean tells us what the best single-valued guess is and the coefficient of variance
    # helps measure uncertainty (how much do the sample networks disagree)
    def test_predictive(self, test_set):
        # Getting the std and mean to compute correctly was too hard with batches, and
        # batch_size=1 is too slow. So, why not just load the whole dataset into VRAM at once?
        # Works great for MNIST on my GPU...
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), pin_memory=True)
        with torch.no_grad():
            dataset_in, dataset_out = next(iter(dataloader))
            dataset_in, dataset_out = dataset_in.to(self.device, non_blocking=True), dataset_out.to(self.device, non_blocking=True)
            
            # Loop through all inputs, computing the predictive distribution for each input
            # and some statistics 
            total_loss = 0
            correct = 0
            total_cfcnt_of_var = 0
            for i in range(dataset_in.size()[0]):
                data, target = dataset_in[i], dataset_out[i].expand((1))
                # print(data)
                # print(target)
                mean, cfcnt_of_var = self.predictive(data)
                # print(mean)
                # print(cfcnt_of_var)
                total_loss += F.nll_loss(mean, target, reduction='sum').item()  # sum up batch loss
                total_cfcnt_of_var += cfcnt_of_var.sum().item()
                pred = mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # print(pred)
                correct += pred.eq(target.view_as(pred)).sum().item()
                # break

        avg_loss = total_loss / float(len(dataloader.dataset))
        avg_cfcnt_of_var = total_cfcnt_of_var / float(len(dataloader.dataset))

        print('\nTest set: Average loss: {:.4f}, Average coefficient of variance: {} Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, avg_cfcnt_of_var, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))
