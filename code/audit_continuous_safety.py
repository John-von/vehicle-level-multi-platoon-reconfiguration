"""Independent continuous all-pair audit from saved states and held inputs.

No controller function is imported. For each quadratic relative trajectory,
enumerate endpoints, stationary points, absolute-value kinks and branch
intersections of max(|dx|-Dx, |dy|-Dy).
"""
import hashlib, json
import numpy as np


def roots(a,b,c):
    """Vectorized real quadratic roots, with independent linear handling."""
    a,b,c=np.broadcast_arrays(a,b,c);out=np.full((len(a),2),np.nan)
    linear=abs(a)<1e-13;ok=linear&(abs(b)>1e-13);out[ok,0]=-c[ok]/b[ok]
    disc=b*b-4*a*c;ok=~linear&(disc>=0)
    sq=np.sqrt(np.maximum(disc[ok],0));aa=a[ok];bb=b[ok]
    out[ok,0]=(-bb-sq)/(2*aa);out[ok,1]=(-bb+sq)/(2*aa)
    return out


def candidates(p,v,a,h,threshold):
    n=len(p);parts=[np.zeros((n,1)),np.full((n,1),h)]
    with np.errstate(divide='ignore',invalid='ignore'):
        parts.extend([(-v[:,d]/a[:,d])[:,None] for d in range(2)])
    for d in range(2):parts.append(roots(.5*a[:,d],v[:,d],p[:,d]))
    for sx in [-1,1]:
        for sy in [-1,1]:
            parts.append(roots(.5*(sx*a[:,0]-sy*a[:,1]),sx*v[:,0]-sy*v[:,1],
                               sx*p[:,0]-sy*p[:,1]-threshold[0]+threshold[1]))
    ts=np.concatenate(parts,axis=1);valid=np.isfinite(ts)&(ts>=0)&(ts<=h)
    ts=np.where(valid,ts,0.)
    xy=p[:,None,:]+v[:,None,:]*ts[:,:,None]+.5*a[:,None,:]*ts[:,:,None]**2
    margin=np.max(abs(xy)-threshold,axis=2);margin[~valid]=np.inf
    k=np.argmin(margin,axis=1)
    return margin[np.arange(n),k],ts[np.arange(n),k]


def conditional_min(p,v,a,h,condition_dim,bound,target_dim):
    """Infimum of |target| on the closure of |condition| < bound."""
    n=len(p);parts=[np.zeros((n,1)),np.full((n,1),h)]
    with np.errstate(divide='ignore',invalid='ignore'):
        parts.append((-v[:,target_dim]/a[:,target_dim])[:,None])
    parts.append(roots(.5*a[:,target_dim],v[:,target_dim],p[:,target_dim]))
    for sign in [-1,1]:parts.append(roots(.5*a[:,condition_dim],v[:,condition_dim],p[:,condition_dim]-sign*bound))
    ts=np.concatenate(parts,axis=1);valid=np.isfinite(ts)&(ts>=0)&(ts<=h);ts=np.where(valid,ts,0)
    xy=p[:,None,:]+v[:,None,:]*ts[:,:,None]+.5*a[:,None,:]*ts[:,:,None]**2
    valid&=abs(xy[:,:,condition_dim])<=bound+1e-9
    val=abs(xy[:,:,target_dim]);val[~valid]=np.inf;k=np.argmin(val,axis=1)
    return val[np.arange(n),k],ts[np.arange(n),k]


def state_at(x,u,k,t):
    s=x[k].copy();s[:,:2]+=s[:,2:]*t+.5*u[k]*t*t;s[:,2:]+=u[k]*t
    return s


def audit(folder,out):
    """Audit of one recorded run; the report and the margin curves are written to the folder out."""
    s=json.loads((folder/'summary.json').read_text(encoding='utf-8'));sc=json.loads((folder/'scenario.json').read_text(encoding='utf-8'))
    z=np.load(folder/'trajectory.npz');x,u=z['trajectory'],z['controls'];c=s['config'];h=c['dt'];n=len(x[0])
    ii,jj=np.triu_indices(n,1);body=np.array([c['length']+c['width']*np.sin(c['heading_max']),c['width']+c['length']*np.sin(c['heading_max'])])
    safety=np.array([c['safety_dx'],c['safety_dy']]);best={k:dict(value=float('inf')) for k in ['safety','body','conditional_x','conditional_y','overlap_x']}
    series={k:[] for k in best};violations=dict(safety=0,body=0)
    for k in range(len(u)):
        p=x[k,ii,:2]-x[k,jj,:2];v=x[k,ii,2:]-x[k,jj,2:];a=u[k,ii]-u[k,jj]
        result={'safety':candidates(p,v,a,h,safety),'body':candidates(p,v,a,h,body),
                'conditional_x':conditional_min(p,v,a,h,1,safety[1],0),
                'conditional_y':conditional_min(p,v,a,h,0,safety[0],1),
                'overlap_x':conditional_min(p,v,a,h,1,body[1],0)}
        for key,(values,tau) in result.items():
            ix=int(np.argmin(values));val=float(values[ix]);series[key].append(val)
            if key in violations:violations[key]+=int(np.sum(values < -1e-9))
            if val<best[key]['value']:
                st=state_at(x,u,k,tau[ix]);i,j=int(ii[ix]),int(jj[ix]);delta=abs(st[i,:2]-st[j,:2])
                best[key]=dict(value=val,step=k,tau=float(tau[ix]),time=k*h+float(tau[ix]),pair=[i,j],
                               labels=[sc['labels'][i],sc['labels'][j]],delta=delta.tolist(),
                               speeds=[float(st[q,2]+c['cruise']) for q in [i,j]])
    expected=np.concatenate([x[:-1,:,:2]+h*x[:-1,:,2:]+.5*h*h*u,x[:-1,:,2:]+h*u],axis=2)
    maxdyn=float(abs(x[1:]-expected).max());heading=abs(np.arctan2(x[:,:,3],x[:,:,2]+c['cruise']))
    half=body[1]/2;road=1e9
    for k in range(len(u)):
        p=x[k,:,1];v=x[k,:,3];a=u[k,:,1]
        with np.errstate(divide='ignore',invalid='ignore'):tau=-v/a
        tau=np.where(np.isfinite(tau)&(tau>0)&(tau<h),tau,0.)
        ys=np.array([p,p+h*v+.5*h*h*a,p+tau*v+.5*tau*tau*a])
        road=min(road,float(min(ys.min()-half+c['lane_width']/2,2.5*c['lane_width']-ys.max()-half)))
    du=np.diff(np.concatenate([np.zeros_like(u[:1]),u]),axis=0)/h
    bounds=dict(speed_min=float((x[:,:,2]+c['cruise']).min()),speed_max=float((x[:,:,2]+c['cruise']).max()),
                lateral_speed_max=float(abs(x[:,:,3]).max()),ax_max=float(abs(u[:,:,0]).max()),ay_max=float(abs(u[:,:,1]).max()),
                discrete_jx_max=float(abs(du[:,:,0]).max()),discrete_jy_max=float(abs(du[:,:,1]).max()),heading_max=float(heading.max()))
    limits_ok=(bounds['speed_min']>=c['v_min']-5e-4 and bounds['speed_max']<=c['v_max']+5e-4 and
               bounds['lateral_speed_max']<=c['vy_max']+5e-4 and bounds['ax_max']<=c['ax_max']+5e-4 and
               bounds['ay_max']<=c['ay_max']+5e-4 and bounds['discrete_jx_max']<=c['jx_max']+5e-4 and
               bounds['discrete_jy_max']<=c['jy_max']+5e-4 and bounds['heading_max']<=c['heading_max']+5e-4)
    min_bumper=best['overlap_x']['value']-body[0]
    report=dict(case=folder.name,seed=s['seed'],policy=s['policy'],completion=s['success'],duration=s['time'],
                vehicles=n,pairs=len(ii),intervals=len(u),pair_intervals=len(ii)*len(u),
                violations=violations,minima=best,body_envelope_dimensions=body.tolist(),minimum_overlap_bumper_bound=min_bumper,
                road_margin=road,dynamics_residual=maxdyn,bounds=bounds,limits_ok=bool(limits_ok),
                passed=bool(maxdyn<1e-9 and heading.max()<=c['heading_max']+5e-4 and limits_ok and road>0 and
                            best['safety']['value']>=-1e-9 and best['body']['value']>0),
                input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [folder/'trajectory.npz',folder/'summary.json',folder/'scenario.json']})
    (out/(folder.name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/(folder.name+'_curves.npz'),time=np.arange(len(u))*h,**series)
    print(folder.name,report['passed'],'minimum center-buffer margin',best['safety']['value'],'body',best['body']['value'],flush=True)
    return report


def smoke_tests():
    # Both endpoints have 10 m gap, but midpoint x=0. A sampled-only test fails.
    p=np.array([[10.,0.]]);v=np.array([[-100.,0.]]);a=np.array([[500.,0.]])
    values,t=candidates(p,v,a,.4,np.array([8.,2.8]));assert values[0]<0 and 0<t[0]<.4
    # At an active-axis intersection the minimum is between endpoints.
    p=np.array([[9.,2.]]);v=np.array([[-2.,2.]]);a=np.zeros((1,2))
    values,t=candidates(p,v,a,1.,np.array([8.,2.8]));assert abs(t[0]-.45)<1e-12 and abs(values[0]-.1)<1e-12
    # Random dense sampling must never find a lower margin than enumeration.
    rng=np.random.default_rng(137);p=rng.normal(size=(30,2))*10;v=rng.normal(size=(30,2))*5;a=rng.normal(size=(30,2))*3
    val,_=candidates(p,v,a,.4,np.array([8.,2.8]));t=np.linspace(0,.4,4001)
    dense=np.max(abs(p[:,None]+v[:,None]*t[None,:,None]+.5*a[:,None]*t[None,:,None]**2)-[8.,2.8],axis=2).min(axis=1)
    assert np.all(val<=dense+1e-10) and np.all(dense-val<.003)


if __name__=='__main__':
    smoke_tests();print('smoke tests passed')
