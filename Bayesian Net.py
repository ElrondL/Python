# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:11:23 2020

@author: Jiaqi Li
"""
import numpy as np

T = 0
F = 1 #set True = 0 because inpython indexing 0 comes before 1

"""Bayesian net info"""

p_c = np.array([0.5,0.5])
p_r_g_c = np.array([[0.8 , 0.2], [0.2 , 0.8]])
p_s_g_c = np.array([[0.1 , 0.9], [0.5 , 0.5]])#c is rows
p_w_g_sr = np.ndarray([2,2,2]) #first 2x2 array is for s = true
p_w_g_sr[T,T,:] = ([0.99, 0.01])
p_w_g_sr[T,F,:] = ([0.9, 0.1])
p_w_g_sr[F,T,:] = ([0.9, 0.1])
p_w_g_sr[F,F,:] = ([0.0, 1.0])

"""Enumeration"""
#Mathematics: p(C=T g. W=T)=p(C,W)/p(W=T)

p_csrw = np.zeros((2,2,2,2), float) #p(C,W) with all combinations
for c in (T,F):
    for s in (T,F):
        for r in (T,F):
            for w in (T,F):
                p_csrw[c,s,r,w] = p_c[c]*p_s_g_c[c,s]*p_r_g_c[c,r]*p_w_g_sr[s,r,w]

p_w = np.sum(p_csrw, axis = (0 ,1 ,2)) #w is last index of p_csrw, so p(w) is 1x2 and sums all columns of p_csrw

p_c_g_w = np.sum(p_csrw, axis = (1,2)) / p_w

print ('Enumeration yields p(c=T | w=T) = %.3f.' % ( p_c_g_w [T,T]))

"""Logistic/Ancestral Sampling (Monte Carlo)"""

samplingRound = 35000
rejected = 0 
trueSamples = 0

for i in range(samplingRound):
    
    #sample cloudy
    s_c = np.random.choice([T,F], p = p_c)
    #sample rain/sprinkler given cloudy
    s_r_g_c = np.random.choice([T,F], p = [p_r_g_c[s_c, T], p_r_g_c[s_c, F]])
    s_s_g_c = np.random.choice([T,F], p = [p_s_g_c[s_c, T], p_s_g_c[s_c, F]])
    #sample wet given rain, sprinkler
    s_w_g_sr = np.random.choice([T,F], p = [p_w_g_sr[s_s_g_c, s_r_g_c, T],p_w_g_sr[s_s_g_c, s_r_g_c, F]])
    
    if s_w_g_sr == T:
        if s_c == T:
            trueSamples += 1
    else:
        rejected += 1 #reject the sample if the case is not wet = T & cloudy = T
    
    num_samples = samplingRound - rejected #total number of samples

print ('Logistic sampling yields p(c=T | w=T) = %.3f.' % (trueSamples/(num_samples)))
print ('(... with %d rejections .)' % ( rejected ))

"""Gibbs Sampling, c is conditionally independent from w"""

p_csw = np.sum(p_csrw, axis = (2)) #p(C,S,W) with all combinations
p_sr = np.sum(p_csrw, axis = (0,3)) #p(S,R)
p_csr = np.sum(p_csrw, axis = (3)) #p(C,S,R)
p_crw = np.sum(p_csrw, axis = (1)) #p(C,R,W)

p_r_g_csw = np.zeros ((2, 2, 2, 2) , float)
for c in (T , F ):
    for s in (T , F ):
        for r in (T , F ):
            for w in (T , F ):
                    p_r_g_csw[c , s , w, r] = p_csrw[c , s , r , w]/p_csw[c,s,w]

p_c_g_sr = np.zeros ((2, 2, 2) , float)
for c in (T , F ):
    for s in (T , F ):
        for r in (T , F ):
            for w in (T , F ):
                    p_c_g_sr[s, r, c] = p_csr[c , s , r]/p_sr[s, r]

p_s_g_crw = np.zeros ((2, 2, 2, 2) , float)
for c in (T , F ):
    for s in (T , F ):
        for r in (T , F ):
            for w in (T , F ):
                    p_s_g_crw[c , r , w, s] = p_csrw[c , s , r , w]/p_crw[c,r,w]
#the conditionals,no conditional for wet (we know wet is true)
#note these conditionals must be constructed with for loop specifying index because the condition probablities are not vectors like p_w

samplingRound2 = 100000
count = 0
s_w = T
s_r = s_s = T #initializer

for s in range(samplingRound2):
    s_c = np.random.choice([T , F], p = [p_c_g_sr[s_s, s_r, T], p_c_g_sr[s_s, s_r, F]]) #draw c sample based on random s, r and w observation
   
    s_r = np.random.choice([T , F], p = [p_r_g_csw [s_c , s_s , s_w, T], p_r_g_csw [s_c , s_s , s_w, F]])
    s_s = np.random.choice([T , F], p = [p_s_g_crw [s_c , s_r , s_w, T], p_s_g_crw [s_c , s_r , s_w, F]]) #set new predictor for r and s

    if s_c == T:
        count += 1
        
print ('Gibbs sammpling yields p(c=T | w=T) = %.3f.' % (count / samplingRound2))
print ('(... with %d samples .)' % (count))




                

