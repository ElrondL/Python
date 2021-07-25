## Implements Metropolis - Hastings

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

# target distribution mixture components (do this here for efficiency)
MIX = []
MU1 = 20 
SIGMA1 = 3
MIX.append(norm(MU1, SIGMA1))
MU2 = 40
SIGMA2 = 10
MIX.append(norm(MU2, SIGMA2))
W = 1.0/len(MIX)

def evalTarget(x):
    ## Evaluates the (often unnormalised) target distribution, p.
    ## Here a mixture of Gaussians.

    #Compute weighted sum of all components
    p_x = 0.0
    for component in MIX:
        p_x += W * component.pdf(x)
        return p_x

def evalProposal(x, mu, std):
    ##Evaluates the proposal distribution q at x.
    ##Here an isotropic Gaussian with mean mu and standard deviation std.
    #Evaluate proposal distribution

    q_x = norm(mu, std).pdf(x)

    return q_x
    
def sampleFromProposal(mu, std, no_samples = 1):
    ##Samples from the proposal distribution q.
    ##Here an isotropic Gausian with mean mu and standard deviation std.
    #Evaluate proposal distribution

    x = norm(mu, std).rvs(no_samples)
    return float(x)

def main():
    #number of samples
    N = 2000
    #standard deviation of proposal
    std = 10.0

    #initialise at mode of target distribution (might have found this using optimisation)
    x_old = MU1
    samples = []
    accepted = 0.0

    #lets get sampling...
    #(note, for this example we will not be sub-sampling the chain)
    for i in range(N):
        #produce some status output
        if i % 200 == 0:
            print ("...running iteration %d..." % i)

        #draw sample from proposal distribution
        x = sampleFromProposal(x_old,std)

        #generate random number from uniform distribution
        z = np.random.uniform(0, 1.0, 1)

        #compute A
        A = min(1.0, (evalTarget(x) * evalProposal(x_old, x, std))/(evalTarget(x_old)*evalProposal(x,x_old,std)))

        #accept or reject new sample
        if z < A:
            samples.append(x)
            x_old = x
            accepted += 1
        else:
            samples.append(x_old)

    #print stats
    print("Acceptance rate: %f%%" % ( accepted / N * 100))

    #show actual target distribution
    state_space = np.linspace(0,100,10000)
    p_x = evalTarget(state_space)

    fig = plt.figure(1, figsize=(15,10))
    ax1 = fig.add_subplot(121, projection = '3d')
    ax1.plot(N*np.ones(len(state_space)), state_space, p_x, 
             linewidth = 2.0, label='Actual Distribution p(x)')
    ax1.plot(range(N), samples, np.zeros(N), label = 'Samples Drawn')
    ax1.set_xlabel('sample number')
    ax1.set_ylabel('state space (x)')
    ax1.set_zlabel('p(x)')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(state_space, p_x, linewidth = 2.0,
             label = 'Actual Distribution p(x)')
    ax2.hist(samples,50,normed=1, alpha= 0.75,
             label ='Sampled Distribution p(x)')
    ax2.set_xlabel('state space (x)')
    ax2.set_ylabel('p(x)')
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    main()