import numpy as np

#for readability, let's define true and false states
T = 0
F = 1

p_c = np.array([0.5, 0.5])
p_s_given_c = np.array([[0.1, 0.5], [0.9,0.5]])
p_r_given_c = np.array([[0.8,0.2],[0.2, 0.8]])
p_w_given_sr = np.ndarray((2,2,2))
p_w_given_sr[T, T, :] =  ([0.99,0.9])
p_w_given_sr [T , F , :] = ([0.9 , 0.0])
p_w_given_sr [F , T , :] = ([0.01 , 0.1])
p_w_given_sr [F , F , :] = ([0.1 , 1.0])

 # set up some other storage
p_c_given_w = np . zeros ((2 ,2) , float )
p_w = np . zeros ((2) , float )
p_csrw = np . zeros ((2 , 2, 2, 2) , float )


# ----------- Compute p(c = T | w = T) via enumeration -----------

# p( wcrs ) = p(c) p(r|c)p(s|c)p(w|rs)
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_csrw [ c_state , s_state , r_state , w_state ] = p_c [ c_state ]\
                * p_r_given_c [ r_state , c_state ] \
                * p_s_given_c [ s_state , c_state ] \
                * p_w_given_sr [ w_state , s_state , r_state ]
 
# p(w) = sum_ (rsc ) p(w, c, r, s)
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_w [ w_state ] = p_w [ w_state ] + p_csrw [ c_state , s_state , r_state , w_state ]


# note : same as
# p_w = np. sum ( p_csrw , axis = (0 ,1 ,2))

# p(c | w) = sum_ {r,s}p(w, c, r, s) / p(w) = p(w,c) / p(w) = p(w|c) p(c) / p(w)
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_c_given_w [ c_state ][ w_state ] = p_c_given_w [ c_state , w_state ] \
                + p_csrw [ c_state , s_state , r_state , w_state ] \
                / p_w [ w_state ]

# note : same as
# p_c_given_w = np. sum ( p_csrw , axis = (1 ,2)) / p_w

print ' Enumeration yields p(c=T | w=T) = %.3f.' % ( p_c_given_w [T , T ])

# ----------------------------------------------------------------


# ----------- Compute p(c = T | w = T) via ancestral sampling ( actually logic sampling ) -----------

# this being ancestral sampling we have to start at the root and go forward along the arrows ...

num_runs = 10000
rejected = 0
c_true_count = 0
for s in range ( num_runs ):
# sample root node : cloudy
    s_c = np . random . choice ([ T , F] , p = p_c )

# sample rain given cloudy
    s_r = np . random . choice ([ T , F] , p =[ p_r_given_c [T , s_c ], p_r_given_c [F , s_c ]])

# sample sprinkler given cloudy
    s_s = np . random . choice ([ T , F] , p =[ p_s_given_c [T , s_c ] , p_s_given_c [F , s_c ]])

# sample wet given rain and sprinkler
    s_w = np . random . choice ([ T , F] , p =[ p_w_given_sr [T , s_s , s_r ], p_w_given_sr [F , s_s , s_r ]])

# now need to make sample fit observation . So if s_w is not True we shall discard the whole sample
# ... of course , here we only care about the state of cloudy , so let ’s keep a record of that rather
# ... than store all things
if s_w == T :
    if s_c == T :
        c_true_count += 1.0
    else :
        rejected += 1.0

# note that the number of samples is NOT just the number of runs but we have to
# account for the rejected ones as well
num_samples = num_runs - rejected
print (’Logic sampling yields p(c=T | w=T) = %.3 f.’ % ( c_true_count / num_samples ))
print (’(... with %d rejections .) ’ % ( rejected ))

# ----------------------------------------------------------------

# ----------- Compute p(c = T | w = T) via Gibbs sampling -----------

# ok , let ’s start by computing the conditionals we will need to sample from ...
# ... in doing this , let ’s re - use the joint distribution computed in part (a)


# p(c | r,s,w)
# as c is conditionally independent of w given r and s (you can read this from the graph
# or show it numerically # when considering sampling requires " observing " r and s),
# this can be computed as
# p(c | r,s) = p(r,s,c) / p(r,s)
p_rsc = np . zeros ((2 , 2, 2) , float )
p_rs = np . zeros ((2 , 2) , float )
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_rs [ r_state , s_state ] = p_rs [ r_state , s_state ] \
                + p_csrw [ c_state , s_state , r_state , w_state ]
                p_rsc [ r_state , s_state , c_state ] = p_rsc [ r_state , s_state , c_state ] \
                + p_csrw [ c_state , s_state , r_state , w_state ]


p_c_given_rs = np . zeros ((2 , 2, 2) , float )
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_c_given_rs [ c_state , r_state , s_state ] = p_rsc [ r_state , s_state , c_state ] \
                / p_rs [ r_state , s_state ]


# p(r | c,s,w) = p(r,s,w,c) / p(c,s,w)
p_csw = np . zeros ((2 , 2, 2) , float )
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_csw [ c_state , s_state , w_state ] = p_csw [ c_state , s_state , w_state ] \
                + p_csrw [ c_state , s_state , r_state , w_state ]
p_r_given_csw = np . zeros ((2 , 2, 2, 2) , float )
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
            p_r_given_csw [ r_state , c_state , s_state , w_state ] = p_csrw [ c_state , s_state , r_state , w_state ] \
            / p_csw [ c_state , s_state , w_state ]

# p(s | c,r,w) = p(r,s,w,c) / p(c,r,w)
p_crw = np . zeros ((2 , 2, 2) , float )
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_crw [ c_state , r_state , w_state ] = p_crw [ c_state , r_state , w_state ] \
                + p_csrw [ c_state , s_state , r_state , w_state ]

p_s_given_crw = np . zeros ((2 , 2, 2, 2) , float )
for c_state in (T , F ):
    for s_state in (T , F ):
        for r_state in (T , F ):
            for w_state in (T , F ):
                p_s_given_crw [ s_state , c_state , r_state , w_state ] = p_csrw [ c_state , s_state , r_state , w_state ] \
                / p_crw [ c_state , r_state , w_state ]

# of course , we have observed w - so do not need to sample that . No conditional needed .
# now , let ’s do sampling ...
num_runs = 10000
c_true_count = 0
# we know the grass is wet ( because we are told we observe it ...
s_w = T
# let ’s arbitrarily initialise the rest (we will , of course , only sample over these )
s_c = s_r = s_s = T
for s in range ( num_runs ):
    # arbitrarily , let ’s start at the top
    s_c = np . random . choice ([ T , F] , p = [ p_c_given_rs [T , s_r , s_s ], p_c_given_rs [F , s_r , s_s ]])
    # ... and move on
    s_r = np . random . choice ([ T , F] , p = [ p_r_given_csw [T , s_c , s_s , s_w ], p_r_given_csw [F , s_c , s_s , s_w ]])
    s_s = np . random . choice ([ T , F] , p =[ p_s_given_crw [T , s_c , s_r , s_w ], p_s_given_crw [F , s_c , s_r , s_w ]])

# as before , we only care about the state of cloudy , so let ’s keep a record of that rather
# ... than store all things ( note also , that here we do not discard any samples , unlike before )
if s_c == T :
    c_true_count += 1.0

print 'Gibbs sammpling yields p(c=T | w=T) = %.3 f.' % ( c_true_count / num_runs )

