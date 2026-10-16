#%%
import time
import numpy as np
from fit_joint_mocks import log_probability, theta0, FIT_DATA, FIT_KEYS
from common import model_multipoles, ap_params

def timeit(f, n=200):
    f()                                           # warm-up
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n * 1e3    # ms

d = FIT_DATA[FIT_KEYS[0]]
h, ombh2, omch2, lnAs, ns = theta0[:5]
theta7 = [h, ombh2, omch2, lnAs, 1.0, 0.0, ns]           # model_multipoles order
emu_p = [1.0, 0.0, ombh2, omch2, h, lnAs, ns]            # emulator order

print(f"joint log_probability (5 tracers): {timeit(lambda: log_probability(theta0)):.1f} ms")
print(f"one-tracer model_multipoles:       {timeit(lambda: model_multipoles(theta7, d['emus'], d['z_eff'], d['s_data'])):.1f} ms")
print(f"  emulators only (3 NNs):          {timeit(lambda: [e(emu_p) for e in d['emus']]):.2f} ms")
print(f"  CAMB ap_params only:             {timeit(lambda: ap_params(h, ombh2, omch2, ns, d['z_eff'])):.1f} ms")
# %%
