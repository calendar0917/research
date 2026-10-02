"""Three bounded upstream hypotheses. NumPy reference; no performance claim.

Extend semantics before the same task dictionary, expose its native code
moments, or regularize its atom values by a shared training mask. Neither
branch propagates a learned state between graph vertices.
"""
from collections import deque
import numpy as np


def shell_semantics(atoms,edges):
    """Unstandardized Sem110 and new shell3 block (28+4+4=36)."""
    n=len(atoms);adj=[[] for _ in atoms]
    for u,v,_ in edges:adj[u].append(v);adj[v].append(u)
    old=np.zeros((n,110),dtype=np.int16);extra=np.zeros((n,36),dtype=np.int16)
    pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    for root in range(n):
        d=[n+1]*n;d[root]=0;q=deque([root])
        while q:
            u=q.popleft()
            if d[u]>=3:continue
            for v in adj[u]:
                if d[v]==n+1:d[v]=d[u]+1;q.append(v)
        for u,a in enumerate(atoms):
            if d[u]<=2:old[root,d[u]*28+a]+=1
            elif d[u]==3:extra[root,a]+=1
        for u,v,b in edges:
            a,c=sorted((d[u],d[v]))
            if c<=2:old[root,84+pairs.index((a,c))*4+b]+=1
            elif (a,c)==(2,3):extra[root,28+b]+=1
            elif (a,c)==(3,3):extra[root,32+b]+=1
        old[root,108]=sum(z<=2 for z in d)
        old[root,109]=sum(d[u]<=2 and d[v]<=2 for u,v,_ in edges)
    return old,extra


def code_summary(c,graph_id,n_graphs):
    """c_i = rho_i * alpha_i. Two native-atom moments per graph."""
    out=np.zeros((n_graphs,2*c.shape[1]))
    np.add.at(out[:,:c.shape[1]],graph_id,c)
    np.add.at(out[:,c.shape[1]:],graph_id,c*c)
    return out


def masked_decode(c,values,mask,p=.1,training=True):
    """One dictionary-atom mask shared by rows; eval reproduces the parent."""
    if not training:return c@values
    return (c*mask[None,:]/(1-p))@values
