#!/usr/bin/env python3
"""Order- and partition-independent checksums of a run's final solution.

Every quantity here is a global reduction, so it does not depend on how the mesh
is numbered or divided. sum(h*area) is the water volume and is INVARIANT under a
permutation of values among cells; the weighted moments are not, which is the
point.
"""
import sys, glob, os, h5py, numpy as np, xml.etree.ElementTree as ET
def fields(d):
    fs=sorted(glob.glob(os.path.join(d,'output','*.xmf')), key=lambda p:int(p.split('-')[-1].split('.')[0]))
    root=ET.parse(fs[-1]).getroot(); base=os.path.dirname(os.path.abspath(fs[-1]))
    out={}
    for a in root.findall('.//Attribute'):
        fn,ds=a.find('DataItem').text.strip().split(':')
        with h5py.File(os.path.join(base,fn),'r') as f: out[a.get('Name')]=f[ds][...]
    return out
def moments(d):
    f=fields(d); h,A,x,y=f['Height'],f['Area'],f['XC'],f['YC']
    return {'cells':len(h), 'sum h*A':float((h*A).sum()), 'sum h^2*A':float((h*h*A).sum()),
            'sum h*x*A':float((h*x*A).sum()), 'sum h*y*A':float((h*y*A).sum()), 'max h':float(h.max())}
a,b=moments(sys.argv[1]),moments(sys.argv[2])
print(f"  {'quantity':<12}{'serial':>22}{'parallel':>22}{'rel diff':>12}")
for k in a:
    if k=='cells': print(f"  {k:<12}{a[k]:>22}{b[k]:>22}{'same' if a[k]==b[k] else 'DIFFER':>12}"); continue
    r=abs(a[k]-b[k])/max(abs(a[k]),1e-300)
    print(f"  {k:<12}{a[k]:>22.15e}{b[k]:>22.15e}{r:>12.2e}")
