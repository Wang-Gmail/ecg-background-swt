"""Offline inference. No reference, injection mask or annotation is accepted."""
import numpy as np
import pywt
from scipy import ndimage, signal

FS=360
def denoise_ecg(y,fs=360,alpha=3.,c=2.):
    """Return a new 1-D mV array using the frozen offline background method."""
    y=np.asarray(y,dtype=float)
    if fs!=360:raise ValueError('Resample the reference and input consistently to 360 Hz before calling.')
    if y.ndim!=1 or len(y)<360 or not np.all(np.isfinite(y)):
        raise ValueError('A finite one-dimensional ECG with at least 360 samples is required.')
    if alpha<0 or c<0:raise ValueError('alpha and c must be nonnegative.')
    return apply(y,features(y),dict(family='background',alpha=alpha,c=c,shrink='firm'))

def mad(x,axis=None):
    center=np.median(x,axis=axis,keepdims=axis is not None)
    return np.median(np.abs(x-center),axis=axis)/.6744897501960817

def features(y,window=1.,q=.2,wavelet='sym4',level=4):
    n=len(y); pad=512
    yp=np.pad(y,(pad,pad),mode='reflect')
    yp=np.pad(yp,(0,(-len(yp))%(2**level)),mode='reflect')
    coeff=pywt.swt(yp,wavelet,level=level)
    w=round(window*FS); locations=np.arange(0,len(yp),round(.25*FS))
    local=[]; background=[]
    for _,d in coeff:
        xp=np.pad(d,(w//2,w-w//2),mode='reflect')
        windows=np.lib.stride_tricks.sliding_window_view(xp,w)[locations]
        s=mad(windows,axis=1)
        local.append(np.interp(np.arange(len(yp)),locations,s))
        background.append(float(np.quantile(s[(locations>=pad)&(locations<pad+n)],q)))
    return dict(coeff=coeff,local=local,background=background,pad=pad,n=n)

def shrink(d,tau,mode='firm'):
    mag=np.abs(d)
    if mode=='soft':return np.sign(d)*np.maximum(mag-tau,0)
    if mode=='hard':return np.where(mag>=tau,d,0)
    return np.sign(d)*np.where(mag>=2*tau,mag,np.maximum(2*(mag-tau),0))

def apply(y,f,p):
    family=p['family']
    if family=='raw':return y.copy()
    if family=='lowpass':return signal.sosfiltfilt(signal.butter(4,p['cutoff'],fs=FS,output='sos'),y)
    if family=='gate_common':
        r=f['local'][-1]/max(f['background'][-1],1e-12)
        gate=ndimage.gaussian_filter1d(np.clip((r-p['ratio'])/.8,0,1),sigma=8)
    new=[]
    for i,(a,d) in enumerate(f['coeff']):
        s=f['local'][i]
        if family=='background':s=np.sqrt(np.maximum(s*s-(p['c']*f['background'][i])**2,0))
        tau=p['alpha']*s
        if family=='gate_common':tau=tau*gate
        new.append((a,shrink(d,tau,p.get('shrink','firm'))))
    return pywt.iswt(new,'sym4')[f['pad']:f['pad']+len(y)]

def historical_gate(y,k=1.2,ratio=1.5):
    n=len(y);pad=256;yp=np.pad(y,(pad,pad),mode='reflect')
    yp=np.pad(yp,(0,(-len(yp))%32),mode='reflect')
    coeff=pywt.swt(yp,'sym4',level=5); local=[]
    centers=np.arange(0,len(yp),90)
    for _,d in coeff:
        sf=[mad(d[max(0,c-90):min(len(d),c+90)]) for c in centers]
        local.append(np.interp(np.arange(len(yp)),centers,sf))
    base=np.percentile(local[-1][pad:pad+n],20)
    gate=ndimage.gaussian_filter1d(np.clip((local[-1]/max(base,1e-9)-ratio)/.8,0,1),sigma=8)
    out=[]
    for i,(a,d) in enumerate(coeff):
        if i<2:out.append((a,d));continue
        tau=k*local[i]*np.sqrt(2*np.log(180))*gate
        out.append((a,d*np.maximum(0,1-(tau/np.maximum(np.abs(d),1e-12))**2)))
    return pywt.iswt(out,'sym4')[pad:pad+n]

def best_variance_split(d,min_len=36):
    """Exact one-change optimization of the literal Eq. 7 absolute-D1 cost."""
    x=np.abs(d);n=len(x);cs=np.r_[0,np.cumsum(x)];sq=np.r_[0,np.cumsum(x*x)]
    t=np.arange(min_len,n-min_len+1);nl=t;nr=n-t
    left=(sq[t]-cs[t]**2/nl)/(nl-1)
    right=(sq[n]-sq[t]-(cs[n]-cs[t])**2/nr)/(nr-1)
    costs=nl*np.log(np.maximum(left,1e-30))+nr*np.log(np.maximum(right,1e-30))
    return int(t[np.argmin(costs)])

def variance_partition(d,intervals,min_len=36,step=9):
    """Optimal fixed-count partition on a declared 25-ms candidate grid."""
    x=np.abs(d);n=len(x);points=np.unique(np.r_[np.arange(0,n,step),n])
    cs=np.r_[0,np.cumsum(x)];sq=np.r_[0,np.cumsum(x*x)]
    length=points[None,:]-points[:,None]
    sums=cs[points][None,:]-cs[points][:,None]
    sums2=sq[points][None,:]-sq[points][:,None]
    safe=np.maximum(length,2)
    variance=(sums2-sums*sums/safe)/(safe-1)
    cost=np.where(length>=min_len,length*np.log(np.maximum(variance,1e-30)),np.inf)
    m=len(points);dp=np.full((intervals+1,m),np.inf);prev=np.full((intervals+1,m),-1,dtype=int);dp[0,0]=0
    for k in range(1,intervals+1):
        for j in range(1,m):
            vals=dp[k-1,:j]+cost[:j,j];i=int(np.argmin(vals));dp[k,j]=vals[i];prev[k,j]=i
    j=m-1;bounds=[n]
    for k in range(intervals,0,-1):j=prev[k,j];bounds.append(int(points[j]))
    return sorted(bounds)

def idt_cache(y,wavelet='bior3.1',level=3,intervals=2):
    n=len(y);pad=512;yp=np.pad(y,(pad,pad),mode='reflect')
    yp=np.pad(yp,(0,(-len(yp))%(2**level)),mode='reflect')
    coeff=pywt.swt(yp,wavelet,level=level);bounds=[]
    for lo in range(0,n,10*FS):
        hi=min(n,lo+10*FS);block=coeff[-1][1][pad+lo:pad+hi]
        if intervals==2:
            split=lo+best_variance_split(block)
            bounds.extend([(lo,split,hi-lo),(split,hi,hi-lo)])
        else:
            cuts=variance_partition(block,intervals)
            bounds.extend((lo+a,lo+b,hi-lo) for a,b in zip(cuts[:-1],cuts[1:]))
    taus=[]
    for _,d in coeff:
        tau=np.zeros(len(yp))
        for lo,hi,block_n in bounds:
            s=np.median(np.abs(d[pad+lo:pad+hi]))/.6754
            tau[pad+lo:pad+hi]=s*np.sqrt(2*np.log(block_n))
        tau[:pad]=tau[pad];tau[pad+n:]=tau[pad+n-1]
        taus.append(tau)
    return dict(coeff=coeff,taus=taus,pad=pad,n=n,wavelet=wavelet,bounds=bounds)

def idt_apply(cache,alpha=1.,mode='hard'):
    coeff=[(a,shrink(d,alpha*tau,mode)) for (a,d),tau in zip(cache['coeff'],cache['taus'])]
    return pywt.iswt(coeff,cache['wavelet'])[cache['pad']:cache['pad']+cache['n']]
