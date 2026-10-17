"""Observation-only matched gates and an attributed author-core NLM port.

NLM follows Brian Tracey's 2013 MATLAB NLM_1dDarbon.m distributed as
File Exchange 41762 v1.1.0. Copyright (c) 2013 Brian Tracey. Original BSD
license and source are retained in ../sources/nlm_author/license.txt.
Python vectorization, scale adapter and tests are project additions.
"""
import numpy as np
import pywt

def gated(y,features,source,shape,alpha,c):
    """All arms share features; only scale source and weight shape change."""
    f=features;new=[]
    if source=='global':
        s0=f['local'][-1];b0=f['background'][-1]
        ratio=np.divide(c*b0,s0,out=np.full_like(s0,np.inf),where=s0>0)
        if c==0:ratio[:]=0
        weight=np.sqrt(np.maximum(1-ratio*ratio,0)) if shape=='sqrt' else np.maximum(1-ratio,0)
    for j,(a,d) in enumerate(f['coeff']):
        s=f['local'][j];b=f['background'][j]
        if source=='global':threshold=alpha*s*weight
        elif shape=='sqrt':threshold=alpha*np.sqrt(np.maximum(s*s-(c*b)**2,0))
        else:threshold=alpha*np.maximum(s-c*b,0)
        mag=np.abs(d)
        z=np.sign(d)*np.where(mag>=2*threshold,mag,np.maximum(2*(mag-threshold),0))
        new.append((a,z))
    return pywt.iswt(new,'sym4')[f['pad']:f['pad']+len(y)]

def nlm_many(y,bandwidths,search=1000,half=10):
    """Author weights exp(-patch SSD/(2*(2*half+1)*lambda^2)).

    Preserves author's zero distance contribution outside shifted bounds,
    strict MATLAB t>1 condition, self weights, and edge exclusion. Edge
    output is filled with the input, outside the experiment's valid margin.
    """
    y=np.asarray(y,dtype=float);n=len(y)
    bandwidths=np.maximum(np.asarray(bandwidths,dtype=float),1e-15)
    h=2*(2*half+1)*bandwidths**2
    centers=np.arange(half+1,n-half)
    accum=np.zeros((len(h),len(centers)));norm=np.zeros_like(accum)
    for offset in range(-search,search+1):
        ssd=np.zeros(n)
        lo=max(0,-offset);hi=min(n,n-offset)
        if hi<=lo:continue
        ssd[lo:hi]=(y[lo:hi]-y[lo+offset:hi+offset])**2
        cs=np.r_[0.,np.cumsum(ssd)]
        distance=np.maximum(cs[centers+half+1]-cs[centers-half],0)
        target=centers+offset;valid=(target>0)&(target<n)
        w=np.exp(-distance[valid][None,:]/h[:,None])
        accum[:,valid]+=w*y[target[valid]][None,:]
        norm[:,valid]+=w
    out=np.repeat(y[None,:],len(h),axis=0)
    out[:,centers]=accum/(norm+np.finfo(float).eps)
    return out

def nlm_literal(y,bandwidth,search=7,half=3):
    """Independent direct patch-sum oracle for small-array verification."""
    y=np.asarray(y);n=len(y);out=y.copy();h=2*(2*half+1)*bandwidth**2
    for i in range(half+1,n-half):
        numerator=0.;denominator=0.
        for offset in range(-search,search+1):
            target=i+offset
            if not 0<target<n:continue
            distance=0.
            for k in range(i-half,i+half+1):
                if 0<=k+offset<n:distance+=(y[k]-y[k+offset])**2
            weight=np.exp(-distance/h)
            numerator+=weight*y[target];denominator+=weight
        out[i]=numerator/(denominator+np.finfo(float).eps)
    return out
