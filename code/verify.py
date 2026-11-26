"""Exhaustive check of the planner on a small instance and a conservative body-clearance bound."""
from itertools import combinations
from dataclasses import replace
import json
import numpy as np
from scenario import Config,make_scenario
from sequence import optimize_joint,optimize,evaluate


def interleavings(m):
    def visit(counts,path):
        if len(path)==3*m:
            yield path;return
        for s in range(3):
            if counts[s]<m:
                nxt=counts.copy();i=s*m+nxt[s];nxt[s]+=1
                yield from visit(nxt,path+[i])
    yield from visit([0,0,0],[])


def test_small_optimizer():
    """Enumerates every assignment and interleaving of a six-vehicle instance and compares with the dynamic program."""
    cfg=replace(Config(),vehicles_per_source=2)
    sc=make_scenario(61,cfg);n=len(sc['labels'])
    best=float('inf');count=0
    for lane0 in combinations(range(n),n//2):
        dest=np.ones(n,dtype=int);dest[list(lane0)]=0
        routed=dict(sc,route=dest)
        conditional=float('inf')
        for order in interleavings(cfg.vehicles_per_source):
            cost=evaluate(routed,order,cfg)['common_objective']
            conditional=min(conditional,cost);best=min(best,cost);count+=1
        exact=optimize(routed,'combined',cfg)
        assert abs(conditional-exact['common_objective'])<1e-9
    joint=optimize_joint(sc,'combined',cfg,beam_width=100000)
    assert not joint['pruning_used']
    assert abs(best-joint['common_objective'])<1e-9
    return dict(candidates=count,brute_optimum=best,joint_value=joint['common_objective'],passed=True)


def geometric_clearance(x,u,cfg):
    """All pairs; lower bound on max(|dx|-Lbox, |dy|-Wbox) on each interval.

    Exact minimum of absolute quadratics is computed independently for x,y.
    max(min(f),min(g)) is conservative but sufficient if positive.
    """
    n=len(x);h=cfg.dt
    ix,iy=np.triu_indices(n,1)
    p=x[ix,:2]-x[iy,:2];v=x[ix,2:]-x[iy,2:];a=u[ix]-u[iy]
    minv=np.minimum(p,p+h*v+.5*h*h*a)
    maxv=np.maximum(p,p+h*v+.5*h*h*a)
    with np.errstate(divide='ignore',invalid='ignore'):
        star=-v/a
    valid=(star>0)&(star<h)&np.isfinite(star)
    ts=np.where(valid,star,0.)
    atstar=p+ts*v+.5*ts*ts*a
    minv=np.minimum(minv,np.where(valid,atstar,np.inf))
    maxv=np.maximum(maxv,np.where(valid,atstar,-np.inf))
    absmin=np.where((minv<=0)&(maxv>=0),0.,np.minimum(abs(minv),abs(maxv)))
    length=cfg.length+cfg.width*np.sin(cfg.heading_max)
    width=cfg.width+cfg.length*np.sin(cfg.heading_max)
    clearance=np.max(absmin-np.array([length,width]),axis=1)
    at=int(np.argmin(clearance))
    return float(clearance[at]),[int(ix[at]),int(iy[at])]


if __name__=='__main__':
    print(json.dumps(test_small_optimizer(),indent=2))
