import os, time
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'     # before anything imports h5py
import emcee

PATH = '/spiffball/cosweeney/data/chains/fits_11par/compact_apfix/compact_dr1_joint_smin20_apfix.h5'
MAX_STEPS = 4000

b = emcee.backends.HDFBackend(PATH, read_only=True)
it0, t0 = b.iteration, time.time()
time.sleep(60)
it1 = b.iteration
rate = (it1 - it0) / ((time.time() - t0) / 60)
print(f'iteration {it1} of {MAX_STEPS}; {rate:.1f} steps/min; '
      f'ETA {(MAX_STEPS - it1) / rate:.0f} min' if rate > 0 else f'iteration {it1}: no progress in 60 s')

lp, tau = b.get_log_prob(), b.get_autocorr_time(tol=0)
q = it1 // 4
print('median lnP by quarter of the chain:', [round(float(np.nanmedian(lp[i * q:(i + 1) * q])), 1) for i in range(4)])
print(f'tau_max = {tau.max():.0f}  (N/tau_max = {it1 / tau.max():.0f});  mean acceptance = {b.accepted.mean() / it1:.2f}')
h = b.get_chain(discard=it1 // 2)[..., 0]
print(f'h over the second half: {h.mean():.4f} +- {h.std():.4f}')