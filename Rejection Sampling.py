# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:31:26 2020

Rejection sampling of a normal sum using uniform distribution proposal

@author: Jiaqi Li
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt

# make p(x)
norm1 = norm(20,3)
norm2 = norm(40,10) #note by 6 sigma rule 99.7 percent of probabilitites for norm1 and norm2 lie in [10,70] (norm1 lies within norm2's domain)

def p(x):
    p = 0.5*norm1.pdf(x) + 0.5*norm2.pdf(x)
    return p

#run rejection sampling
def main():
######initialize part
    x = [] #sample array
    count = 0.0 #acceptance count
    targetcount = 10000 #question asks for 10000 samples
    samplingRound = 55000 #assume an acceptance rate of 20%

#####proposal density and M
    #create proposal density [5, 75] (6 sigma)
    q = uniform(5, 75)
    
    #find M = sup_x(f(x)/q(x))
    index = np.linspace(5, 75, num=10000) #number of index decides precision of samples
    arrM = []
    for id in index:
        arrM.append(p(id)/q.pdf(id))
    M = np.amax(arrM)

####Run rejection sampling
    print("Each iteration contains 1000 rejection sampling runs.")
    for i in range(samplingRound):
        #status output on iteration
        iteration = i/1000
        if (iteration).is_integer():
            print("running Iteration %i" %iteration)
        
        #rejection sampling
        Y = q.rvs() 
        U = uniform(0,1).rvs() #draw a random number Y from q and another random number U from uni[0,1]
        if U < p(Y)/(M*q.pdf(Y)):
            x.append(Y)
            count += 1.0
    
    print("Acceptance rate: %f%%" % ((count/samplingRound)*100))
    if count >= targetcount :
        print("Target count reached")
    else:
        print("Collected %i samples" %count)
    
####Plot part  
    plt.close('all')
    plt.hist(x, 50, density = 1, alpha = 0.75, label = 'samples collected') #sample histogram, normalized (normed has been deprecated)
    
    plt.plot(index, p(index), linewidth = 2.0, label = 'target distribution') #p(x) visualization
    plt.plot(index, M*q.pdf(index), linewidth = 2.0, label = 'proposal distribution')
    plt.legend()
    #plt.set_xlabel('x')
    #plt.set_ylabel('p(x)')
    plt.show()

if __name__ == "__main__":
    main()
    
    

        
            
    
    