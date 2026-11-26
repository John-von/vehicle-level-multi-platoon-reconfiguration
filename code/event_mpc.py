"""Local maneuver activation with refreshed rectangular collision corridors.

The control clock is synchronous; lane-change decisions are vehicle-specific.
Every accepted local QP is checked against all currently shared trajectories
over every continuous hold interval before it replaces a feasible plan.
"""

import osqp
import numpy as np
from scipy import sparse
from time import perf_counter
from dmpc import reference


def interval_absolute_min(p,v,a,h):
    lo=np.minimum(p,p+h*v+.5*h*h*a)
    hi=np.maximum(p,p+h*v+.5*h*h*a)
    with np.errstate(divide='ignore',invalid='ignore'):star=-v/a
    use=np.isfinite(star)&(star>0)&(star<h)
    t=np.where(use,star,0.)
    q=p+t*v+.5*t*t*a
    lo=np.minimum(lo,np.where(use,q,np.inf));hi=np.maximum(hi,np.where(use,q,-np.inf))
    return np.where((lo<=0)&(hi>=0),0.,np.minimum(abs(lo),abs(hi)))


def prediction_clearance(i,states,controls,shared,candidate_u,candidate_pred,cfg):
    ids=np.array([j for j in range(len(states)) if j!=i])
    self_start=np.concatenate([states[i:i+1],candidate_pred[:-1]],axis=0)
    peer_start=np.concatenate([states[ids,None,:],shared[ids,:-1,:]],axis=1)
    p=self_start[None,:,:2]-peer_start[:,:,:2]
    v=self_start[None,:,2:]-peer_start[:,:,2:]
    a=candidate_u[None,:,:]-controls[ids]
    minimum=interval_absolute_min(p,v,a,cfg.dt)
    margin=np.max(minimum-np.array([cfg.safety_dx,cfg.safety_dy]),axis=2)
    return float(margin.min())


def select_neighbors(i,states,controls,shared,cfg,trust=2.):
    ids=np.array([j for j in range(len(states)) if j!=i])
    starts=np.concatenate([states[:,None,:],shared[:,:-1,:]],axis=1)
    p=starts[i,None,:,0]-starts[ids,:,0]
    v=starts[i,None,:,2]-starts[ids,:,2]
    a=controls[i,None,:,0]-controls[ids,:,0]
    absmin=interval_absolute_min(p,v,a,cfg.dt)
    # Omitted cars have a longitudinal separation larger than the QP's
    # per-update x trust region; all are independently checked after solving.
    return ids[np.min(absmin,axis=1)<cfg.safety_dx+trust+.15].tolist()


def target_order_ready(i,states,shared,dest,target_rank,cfg):
    """Check ordered room for every target slot, including absent inserters."""
    peers=np.flatnonzero((dest==dest[i]) & (np.arange(len(states))!=i))
    signs=np.where(target_rank[i]<target_rank[peers],1.,-1.)
    reserved=np.abs(target_rank[i]-target_rank[peers])*(cfg.safety_dx+.015)
    return bool(np.all(signs*(states[i,0]-states[peers,0])>=reserved) and
                np.all(signs[:,None]*(shared[i,None,:,0]-shared[peers,:,0])>=reserved[:,None]))


class EventQP:
    def __init__(self,i,collision_neighbors,coupling_neighbors,cfg,E,F,fixed_x=None):
        self.i=i;self.coll=collision_neighbors;self.couple=coupling_neighbors
        self.fixed_x={} if fixed_x is None else fixed_x
        self.cfg=cfg;self.E=E;self.F=F;H=cfg.horizon
        self.Gx=F[0::4];self.Gy=F[1::4];self.Gv=F[2::4]
        self.Q=np.tile([1.8,18.,.9,3.],H);self.Q[-4:]*=3
        self.R=np.tile([.25,1.],H)
        self.J=np.tile([cfg.input_rate_weight_x,cfg.input_rate_weight_y],H)/cfg.dt**2
        self.D=np.eye(2*H)-np.eye(2*H,k=-2)
        P=F.T@(self.Q[:,None]*F)+np.diag(self.R)+self.D.T@(self.J[:,None]*self.D)
        P+=len(self.couple)*(cfg.coupling_position_weight*self.Gx.T@self.Gx+cfg.coupling_speed_weight*self.Gv.T@self.Gv)
        self.P=sparse.csc_matrix(np.triu(2*P/cfg.qp_objective_scale))
        tan=np.tan(cfg.heading_max)
        heading=np.vstack([F[3::4]-tan*F[2::4],-F[3::4]-tan*F[2::4]])
        self.A=np.vstack([F,np.eye(2*H),self.D,F[-2:],heading,self.Gx,
                          *[G for _ in self.coll for G in (self.Gx,self.Gy)]])
        self.solver=osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(2*P/cfg.qp_objective_scale)),q=np.zeros(2*H),A=sparse.csc_matrix(self.A),
                          l=np.full(len(self.A),-np.inf),u=np.full(len(self.A),np.inf),
                          verbose=False,eps_abs=3e-6,eps_rel=3e-7,max_iter=cfg.qp_max_iter,
                          polishing=False,adaptive_rho=True,warm_starting=True)

    def solve(self,states,previous,ref,shared,controls,ylimits,lateral_intent):
        i=self.i;cfg=self.cfg;H=cfg.horizon
        base=self.E@states[i];b=base.reshape(H,4)
        d0=np.zeros(2*H);d0[:2]=previous[i]
        q=self.F.T@(self.Q*(base-ref[i].reshape(-1)))-self.D.T@(self.J*d0)
        for j in self.couple:
            q+=cfg.coupling_position_weight*self.Gx.T@(b[:,0]-shared[j,:,0]-(ref[i,:,0]-ref[j,:,0]))
            q+=cfg.coupling_speed_weight*self.Gv.T@(b[:,2]-shared[j,:,2]-(ref[i,:,2]-ref[j,:,2]))
        state_lo=np.tile([-np.inf,ylimits[0],cfg.v_min-cfg.cruise,-cfg.vy_max],H)
        state_hi=np.tile([np.inf,ylimits[1],cfg.v_max-cfg.cruise,cfg.vy_max],H)
        limit=np.tile([cfg.ax_max,cfg.ay_max],H)
        limit[-2:]=np.minimum(limit[-2:],[cfg.jx_max*cfg.dt,cfg.jy_max*cfg.dt])
        jerk=np.tile([cfg.jx_max,cfg.jy_max],H)*cfg.dt
        tan=np.tan(cfg.heading_max)
        hb=np.r_[b[:,3]-tan*b[:,2],-b[:,3]-tan*b[:,2]]
        lo=[state_lo-base,-limit,-jerk+d0,-b[-1,2:],np.full(2*H,-np.inf),shared[i,:,0]-b[:,0]-2.]
        hi=[state_hi-base,limit,jerk+d0,-b[-1,2:],tan*cfg.cruise-hb,shared[i,:,0]-b[:,0]+2.]
        for j in self.coll:
            delta=shared[i,:,:2]-shared[j,:,:2]
            xgood=abs(delta[:,0])>=cfg.safety_dx+.005
            ygood=abs(delta[:,1])>=cfg.safety_dy+.005
            # Crossing intentions prefer x corridors; otherwise y corridors
            # leave independent lanes free to adjust longitudinal positions.
            axis=np.where(xgood & (lateral_intent or not np.all(ygood)),0,1)
            axis=np.where(~xgood,1,axis)
            axis=np.where(~ygood,0,axis)
            if j in self.fixed_x:
                axis[:]=0
                delta[:,0]=self.fixed_x[j][0]
            x_threshold=self.fixed_x[j][1] if j in self.fixed_x else cfg.safety_dx+.02
            for dim,threshold in [(0,x_threshold),(1,cfg.safety_dy+.02)]:
                ll=np.full(H,-np.inf);uu=np.full(H,np.inf)
                plus=(axis==dim)&(delta[:,dim]>=0);minus=(axis==dim)&~plus
                ll[plus]=shared[j,plus,dim]+threshold-b[plus,dim]
                uu[minus]=shared[j,minus,dim]-threshold-b[minus,dim]
                lo.append(ll);hi.append(uu)
        l,u=np.concatenate(lo),np.concatenate(hi)
        start=perf_counter()
        if cfg.qp_solver=='clarabel':
            import clarabel
            from types import SimpleNamespace
            equal=np.isfinite(l)&np.isfinite(u)&(abs(u-l)<1e-10)
            lower=np.isfinite(l)&~equal;upper=np.isfinite(u)&~equal
            aa=sparse.csc_matrix(np.vstack([self.A[equal],self.A[upper],-self.A[lower]]))
            bb=np.r_[u[equal],u[upper],-l[lower]]
            settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
            settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-8
            cones=[clarabel.ZeroConeT(int(equal.sum())),clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()))]
            solved=clarabel.DefaultSolver(self.P,2*q/cfg.qp_objective_scale,aa,bb,cones,settings).solve()
            status=str(solved.status)
            result=SimpleNamespace(x=np.array(solved.x),info=SimpleNamespace(status=status,status_val=1 if status=='Solved' else 0,iter=solved.iterations))
        else:
            self.solver.update(q=2*q/cfg.qp_objective_scale,l=l,u=u);self.solver.warm_start(x=controls[i].reshape(-1))
            result=self.solver.solve(raise_error=False)
        seconds=perf_counter()-start
        violation=float('inf');clearance=None;candidate=None;pred=None
        if result.x is not None and np.isfinite(result.x).all():
            ax=self.A@result.x;violation=float(max(0,np.max(l-ax),np.max(ax-u)))
        ok=result.info.status_val in (1,2) and violation<=3e-4
        if ok:
            candidate=result.x.reshape(H,2);pred=(base+self.F@result.x).reshape(H,4)
            clearance=prediction_clearance(i,states,controls,shared,candidate,pred,cfg)
            ok=clearance>=-3e-4
        record=dict(vehicle=i,status=result.info.status,accepted=bool(ok),violation=violation,
                    clearance=clearance,seconds=seconds,iterations=result.info.iter,
                    collision_neighbors=list(self.coll))
        return (candidate,pred,record) if ok else (None,None,record)


def crossing_gate(i,states,shared,source,dest,rank,cfg,freeze_lateral=None):
    """Local full-maneuver screening, refreshed after every committed plan.

    Beyond the shared horizon peers hold their terminal moving-frame state.
    This is only a launch screen; accepted QPs still certify the complete
    receding horizon continuously. The screen is not a future reservation.
    """
    h=cfg.dt;steps=int(np.ceil(cfg.crossing_time/h));times=np.arange(steps+1)*h
    own=reference(states[i:i+1],np.array([[states[i,0],dest[i]*cfg.lane_width]]),cfg.crossing_time,times)[0]
    forecast=np.concatenate([states[:,None,:],shared],axis=1)
    if steps>cfg.horizon:
        forecast=np.concatenate([forecast,np.repeat(forecast[:,-1:,:],steps-cfg.horizon,axis=1)],axis=1)
    else:forecast=forecast[:,:steps+1]
    if freeze_lateral is not None:
        forecast=forecast.copy();forecast[freeze_lateral,:,1]=source[freeze_lateral]*cfg.lane_width
    own[:,0]=forecast[i,:,0]
    dy=abs(own[None,:,1]-forecast[:,:,1]);dx=own[None,:,0]-forecast[:,:,0]
    sign=np.where(rank[i]<rank,1.,-1.)
    signed=sign[:,None]*dx
    # The intermediate interleaving is also a precedence contract. Entering
    # behind a car that must later fall behind us can trap both in one lane,
    # even when their final destinations differ. Preserve the chosen order
    # whenever candidate and peer lateral occupancies overlap.
    goodx=signed>=cfg.safety_dx+.015
    good=goodx|(dy>=cfg.safety_dy+.06)
    blockers=np.flatnonzero(~np.all(good,axis=1)).tolist()
    blockers=[j for j in blockers if j!=i]
    return not blockers,blockers
