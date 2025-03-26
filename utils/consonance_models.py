# Dissonance models
import numpy as np

b1=3.5
b2=5.75

def expdif(f,g):
    """Auxiliary parameter for the models' formulas.
    """
    fmax,fmin=np.maximum(f,g),np.minimum(f,g)
    s=0.24/(0.0207*fmin+18.96)
    return np.exp(-b1*s*(fmax-fmin))-np.exp(-b2*s*(fmax-fmin))

def Sethares1(a,b,f,g):
    return a*b*expdif(f,g)

def Vassilakis(a,b,f,g):
    amin=np.minimum(a,b)
    return 0.5*((a*b)**0.1)*((2*amin/(a+b))**3.11)*expdif(f,g)

def D(F,A,method):
    """Dissonance of a complex tone.
    
    Here, F and A are its frequency and amplitude vectors.
    """
    n=len(A)
    
    if method=='Sethares1':
        d=Sethares1
    elif method=='Vassilakis':
        d=Vassilakis
    
    i,j=np.triu_indices(n,1)
    
    if F.ndim==1:
        dissonances=d(A[i],A[j],F[i],F[j])
        value=np.sum(dissonances)
    elif F.ndim==2:
        dissonances=d(A[i],A[j],F[:,i],F[:,j])
        value=np.sum(dissonances,axis=1)
    return value    

def Dis(F,A,α,method):
    """Dissonance curve for a complex tone.
    
    Here, F and A are its frequency and amplitude vectors 
    and α is the variable-frequency ratio.
    """
    return Dis2(F,A,F,A,α,method)

def Disi(F,A,α,method): 
    return Disi2(F,A,F,A,α,method)

def D2(F1,A1,F2,A2,method):
    """Dissonance of two complex tones.
    
    Here, Fi and Ai are the frequency and amplitude vectors of a complex tone.
    """

    n=len(A1)
    m=len(A2)
    
    if method=='Sethares1':
        d=Sethares1
    elif method=='Vassilakis':
        d=Vassilakis
    
    i,j=np.indices((n,m),sparse=True)
    dissonances=d(A1[i],A2[j],F1[i],F2[j])
    return D(F1,A1,method)+D(F2,A2,method)+np.sum(dissonances)

def Dis2(F1,A1,F2,A2,α,method):
    """Dissonance curve for two complex tones.
    
    Here, Fi and Ai are their frequency and amplitude vectors 
    and α is the variable-frequency ratio.
    """
    
    if method=='Sethares1':
        d=Sethares1
    elif method=='Vassilakis':
        d=Vassilakis
    
    n=len(A1)
    m=len(A2)
    i,j=np.indices((n,m),sparse=True)
    dissonances=d(A1[i],A2[j],F1[i],np.outer(α,F2)[:,[j]])
    return D(F1,A1,method)+D(np.outer(α,F2),A2,method)+np.sum(dissonances,axis=(3,2,1))

def Disi2(F1,A1,F2,A2,α,method):
    if method=='Sethares1':
        d=Sethares1
    elif method=='Vassilakis':
        d=Vassilakis
    
    n=len(A1)
    m=len(A2)
    i,j=np.indices((n,m),sparse=True)
    
    return D(F1,A1,method)+np.array([D(r*F2,A2,method)+np.sum(d(A1[i],A2[j],F1[i],r*F2[j])) for r in α])