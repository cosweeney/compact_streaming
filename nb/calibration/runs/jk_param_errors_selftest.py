"""Self-test for jk_param_errors.py on synthetic Gaussian halos (known answer: sigma_0/sqrt(2 N_halos)); needs only the repo.
   python jk_param_errors_selftest.py --n-mc 20   (expect JK ~ Monte Carlo ~ analytic, kappa ~ 1)"""
import numpy as np, sys, time, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--n-mc',type=int,default=40); K=ap.parse_args().n_mc
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.abspath('.'))   # run from the repo root
import jk_param_errors as jk, param_shot_errors as pse, compact.model as M
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
L=100.; Nh=4000; sig1=300.; nv=300; vlo,vhi=-3000.,3000.; NB=64
R_edges=np.linspace(2,30,8); rl_edges=np.linspace(0,30,8); nR=7
theta=np.array([sig1,0.,100.,30.,0.,50.,30.,1.,0.2,1000.,0.5])
dvv=(vhi-vlo)/nv; vc=vlo+(np.arange(nv)+0.5)*dvv
def relc(p0,pn,L_):
    d=pn-p0; return d-L_*np.round(d/L_)
def total_hist(pos,vel):                     # production semantics: pair counted once, from the lower member (rel z > 0)
    tree=cKDTree(pos,boxsize=L); rq=np.hypot(30,30); C=np.zeros((nR,nR,nv),dtype=np.int64)
    for i in range(len(pos)):
        nb=np.asarray(tree.query_ball_point(pos[i],rq)); nb=nb[nb!=i]
        if nb.size==0: continue
        d=relc(pos[i],pos[nb],L); R=np.hypot(d[:,0],d[:,1]); k=(R<=30)&(R>0)&(np.abs(d[:,2])<=30)
        if not k.any(): continue
        dv=vel[nb[k],2]-vel[i,2]; iR=np.searchsorted(R_edges,R[k],side='right')-1; il=np.searchsorted(rl_edges,d[k,2],side='right')-1
        iv=np.floor((dv-vlo)/(vhi-vlo)*nv).astype(int); ok=(iR>=0)&(iR<nR)&(il>=0)&(il<nR)&(iv>=0)&(iv<nv); np.add.at(C,(iR[ok],il[ok],iv[ok]),1)
    return C
def box(seed):
    rng=np.random.default_rng(seed); return rng.uniform(0,L,(Nh,3)), rng.normal(0,sig1,(Nh,3))
pos,vel=box(1); C=total_hist(pos,vel); n_full=C.sum(-1); sum_v=C@vc
prep=jk.prepare_cell(theta,R_edges,rl_edges,n_full,sum_v,vlo,vhi,nv,M,n_min=50)
block_of=np.random.default_rng(0).permutation(Nh)%NB; groups=[np.where(block_of==r)[0] for r in range(NB)]
t0=time.time(); Mf=np.zeros((NB,NB,11)); Nf=np.zeros((NB,NB))
for r in range(NB): Mf[r],Nf[r]=jk.jk_chunk(pos,vel,groups[r],block_of,NB,prep,L,30.,30.,1.0,relc)
print(f"JK pair search: {time.time()-t0:.1f} s; pair count check: sum N = {Nf.sum():.4e} vs production-style total {n_full[prep['used']].sum():.4e}")
res=jk.jk_from_blocks(prep,Mf,Nf,Nh)
Fbar=prep['F']/prep['N_pairs']; varm=res['varm']; sjk=np.sqrt(varm[0,0])/Fbar[0,0]; sp=prep['F'][0,0]**-0.5; Np=prep['N_pairs']
print(f"sigma0 (others fixed): pair-Fisher {sp:.3f} km/s | JK leave-halos-out {sjk:.3f} km/s | JK/pair {sjk/sp:.1f}, sqrt(Np/Nh) = {np.sqrt(Np/Nh):.1f} -> kappa = {((sjk/sp)**2-1)*Nh/Np:.2f}")
print('mean score per pair (should be ~0 for a correct model):',np.round(res['mean_score'][:4],4))
# Monte Carlo over independent boxes (ML sigma0, others fixed)
RPg,RLg=np.meshgrid(0.5*(R_edges[1:]+R_edges[:-1]),0.5*(rl_edges[1:]+rl_edges[:-1]),indexing='ij'); u=prep['used']; RP,RL=RPg[u],RLg[u]; R3=np.hypot(RP,RL); X=R3-RP
def ml(counts):
    cu=counts[u].astype(float); mu=np.zeros(cu.shape[0])
    def nll(s0):
        p=theta.copy(); p[0]=s0; sg,dl,nu=pse.q_of_p(p,RP,X,R3,M); P=np.maximum(pse.cell_probs(sg,dl,nu,mu,vc,dvv,M),1e-300); return -(cu*np.log(P)).sum()
    return minimize_scalar(nll,bounds=(250,350),method='bounded',options={'xatol':1e-3}).x
est=[]; t0=time.time()
for s in range(2,2+K):
    p_,v_=box(s); est.append(ml(total_hist(p_,v_)))
print(f"Monte Carlo ({len(est)} independent boxes): std(sigma0_hat) = {np.std(est,ddof=1):.3f} km/s  (heuristic sigma_halo = {sp*np.sqrt(Np/Nh):.3f}, analytic sigma0/sqrt(2Nh) = {sig1/np.sqrt(2*Nh):.3f})  [{time.time()-t0:.0f} s]")