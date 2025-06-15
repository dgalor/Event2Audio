#%%
import torch
import timeit
import numpy as np
torch.set_default_device('cuda:0')
#make a sparse matrix (1, n) with k non-zero entries
ns = [100, 1000, 10000]
ps = [0.01, 0.1, 1.0]
times = np.zeros((len(ns), len(ps), 2)) * np.nan
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        k = int(n * p)
        inds = torch.randint(0, n, (k,))
        vals = torch.randn(k)
        v = torch.sparse_coo_tensor(torch.stack([torch.zeros(k), inds]), vals, (1, n))
        v_dense = v.to_dense()
        A = torch.randn(n, n//2)
        t_sparse = timeit.timeit(lambda: torch.sparse.mm(v, A), number=10)
        t_dense = timeit.timeit(lambda: torch.mm(v_dense, A), number=10)
        times[i, j] = [t_sparse, t_dense]
        print(f"n={n}, k={k}, improvement_ratio={t_dense/t_sparse}")
# %%
#now make it batched 
b = 100
times = np.zeros((len(ns), len(ps), 2)) * np.nan
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        k = int(n * p)
        inds = torch.randint(0, n, (b, k))
        vals = torch.randn(b, k)
        batch_ind = torch.arange(b).repeat_interleave(k)
        v = torch.sparse_coo_tensor(torch.stack([batch_ind, inds.view(-1)]), vals.view(-1), (b, n))
        v_dense = v.to_dense()
        A = torch.randn(n, n//2)
        t_sparse = timeit.timeit(lambda: torch.sparse.mm(v, A), number=10)
        t_dense = timeit.timeit(lambda: torch.mm(v_dense, A), number=10)
        times[i, j] = [t_sparse, t_dense]
        print(f"n={n}, k={k}, improvement_ratio={t_dense/t_sparse}")
# %%
