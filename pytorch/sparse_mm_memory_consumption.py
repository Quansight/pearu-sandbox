# Author: Pearu Peterson
# Created: July 2020

import itertools
import torch

def generate_indices(shape):
    for indices in itertools.product(*[list(range(d)) for d in shape]):
        yield indices

def generate_coo_tensor(shape):
    indices_lst = []
    values_lst = []
    for indices in generate_indices(shape):
        indices_lst.append(indices)
        values_lst.append(torch.randn(()))
        yield torch.sparse_coo_tensor(torch.tensor(indices_lst).T, torch.tensor(values_lst), shape).coalesce()


def get_storage_size(a):
    if a.is_sparse:
        return get_storage_size(a._indices()) + get_storage_size(a._values())
    return a.numel() * a.element_size()

def find_sparse_dense_equivalence(N, last_sparsity=None):

    shape = (N,) * 2

    # sparse tensors using about the same amount of memory as dense
    # tensors in the context of matrix multiplication
    same_storage_coalesce = None
    same_storage_uncoalesce = None

    for A in generate_coo_tensor(shape):
        assert A.is_coalesced()
        sparsity = A._values().numel() / A.numel()

        B = A.to_dense()
        AA = torch.sparse.mm(A, A)
        BB = torch.mm(B, B)

        data = dict(
            sparse_mm_storage_uncoalesce = 2*get_storage_size(A) + get_storage_size(AA),
            sparse_mm_storage_coalesce = 2*get_storage_size(A) + get_storage_size(AA.coalesce()),
            dense_mm_storage = 2*get_storage_size(A) + 2*get_storage_size(B) + get_storage_size(BB),
            sparse_vs_dense = get_storage_size(A) / get_storage_size(B),
            sparsity = sparsity
        )
        if same_storage_coalesce is None and data['dense_mm_storage'] < data['sparse_mm_storage_coalesce']:
            data['kind'] = 'coalesce'
            same_storage_coalesce = data
            break
        if same_storage_uncoalesce is None and data['dense_mm_storage'] < data['sparse_mm_storage_uncoalesce']:
            data['kind'] = 'uncoalesce'
            same_storage_uncoalesce = data
    return same_storage_coalesce, same_storage_uncoalesce

for N in [10, 20, 30, 40, 50, 100]:
    coalesce, uncoalesce = find_sparse_dense_equivalence(N)
    print(f'N={N}, coalesce sparsity={coalesce["sparsity"]:.4f}, uncoalesce sparsity={uncoalesce["sparsity"]:.4f}')
