# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:52:33 2020

Metropolis Hasting Sampling of a Normal Sum using Gaussion Distribution Proposal

@author: Jiaqi Li
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits . mplot3d import Axes3D

# make p(x)(target)
norm1 = norm(20,3)
norm2 = norm(40,10) #note by 6 sigma rule 99.7 percent of probabilitites for norm1 and norm2 lie in [10,70] (norm1 lies within norm2's domain)

def p(x):
    p = 0.5*norm1.pdf(x) + 0.5*norm2.pdf(x)
    return p

### functions about q(x)
def q(x, pGaussian):
    q = pGaussian.pdf(x) #function to evaluate proposal distribution at x
    return q

def drawfromq(pGaussian):
    q = pGaussian.rvs() #function to draw random sample from proposal distribution
    return q

def main():
    ###initialization part
    samplingRound = 28000 #assume a 50% ~ 80% acceptance rate 
    
    std = 11 #choose proposal std to be slightly bigger than the target 
    x_0 = 0.5*(20+40) #choose the (weighted) average of the target component gaussian means as proposal mean 
    
    samples = []
    count = 0.0
    targetcount = 10000
    
    ####Run MH sampling
    print("Each iteration contains 2000 MH sampling runs.")
    for i in range(samplingRound):
        
        pGaussian_current = norm(x_0, std) #current proposal distribution
        
        #status output on iteration
        iteration = i/2000
        if (iteration).is_integer():
            print("running Iteration %i" %iteration)
            
        x = drawfromq(pGaussian_current) #draw a random sample
        pGaussian_potential = norm(x, std) #possible future proposal distribution
        
        z = np.random.uniform(0,1) #draw a random number between 0 and 1
        A = min(1.0, (p(x)*q(x_0, pGaussian_potential))/(p(x_0)*q(x, pGaussian_current))) #compute A
        
        if z < A:
            samples.append(x)
            x_0 = x
            count += 1
        else:
            samples.append(x_0)
    
    print("Acceptance rate: %f%%" % ((count/samplingRound)*100))  
    if count >= targetcount :
        print("Target count reached")
    else:
        print("Collected %i samples" %count)
            
    ####Plotting part
    index = np . linspace (-3 , 63 , 10000)
    p_x = p(index) #proposal distribution
    
    fig = plt.figure(1 , figsize =(15 ,10))
    ax1 = fig.add_subplot(1,2,1, projection ='3d')
    ax1.plot(samplingRound*np.ones(len(index)), index , p_x , linewidth =2.0 , label ='Actual Distribution p(x)')
    ax1.plot(range(samplingRound), samples, np.zeros(samplingRound), label ='Samples Drawn')
    ax1.set_xlabel('sample number')
    ax1.set_ylabel('index (x)')
    ax1.set_zlabel('p(x)')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(index, p_x, linewidth =2.0, label ='Actual Distribution p(x)')
    ax2.hist(samples, 50, density =1, alpha =0.75 , label ='Sampled Distribution p(x)')
    ax2.set_xlabel('state space (x)')
    ax2.set_ylabel('p(x)')
    ax2.legend()

    plt.show()
    
if __name__ == "__main__":
    main()

    
    
        
    