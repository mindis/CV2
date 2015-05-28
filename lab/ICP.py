#This function performs Iterative Closest point merging, as explained in 
# "Implementation of a 3D ICP-based Scan Matcher" by Zhongjie and Shou-Yu.
import numpy as np
from scipy.spatial import KDTree

def ICP(base, target):
    geoCenterTarget = np.mean(target, axis=1)
    R = np.eye(3)
    t = np.array([0,0,0])
    
    KDindex = KDTree(target.T)
    lastDistance = np.array([1,1,1])
    
    sample = base
    for i in range(150):
        transformedSample = np.dot(sample.T, R) + t

        I = KDindex.query(transformedSample)
        idx = I[1]
        
        distance = np.sqrt(np.sum(I[0])**2 / transformedSample.shape[0])
        
        if np.sum(np.abs(lastDistance - distance)) < 1e-7:
            break

        lastDistance = distance
        
        geoCenterTransformedSample = transformedSample.mean(axis=0)
        geoCenterTarget = target[:, idx].T.mean(axis = 0)
        
        #Normalize
        A = np.dot( (transformedSample - geoCenterTransformedSample).T, (target[:, idx].T - geoCenterTarget) )
        
        U, s, V = np.linalg.svd(A)
        R = np.dot(U,V)
        geoCentersample = sample.mean(axis=1)
        t = geoCenterTarget - np.dot(geoCentersample, R)
    
    return R, t, distance
