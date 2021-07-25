# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:28:22 2021

@author: ljq-2
"""
import math
from numpy import array
import numpy as np

a, b, c, d = input().split(" ")
a, b, c, d = float(a), float(b), float(c), float(d)

vec1 = array([a,  b])
vec2 = array([c, d])



def angle_of_vectors(vec1, vec2):
    
    dotProduct = vec1.dot(vec2)
    cosofAngle = dotProduct/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    angle = math.acos(cosofAngle)
    print("θ =", angle, "rad") 
    
angle_of_vectors(vec1, vec2)