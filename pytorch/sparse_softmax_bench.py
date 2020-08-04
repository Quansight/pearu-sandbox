
import numpy as np
import time

try:
    import tensorflow as tf
    print('Using tensorflow %s' % (tf.__version__))
except ImportError as msg:
    print('failed to import tensorflow: %s' % (msg))
    tf = None

try:
    import torch as pt
    print('Using pytorch %s' % (pt.__version__))
except ImportError as msg:
    print('failed to import pytorch: %s' % (msg))
    pt = None

np.random.seed(1234)
    
def generate_coo_data(size, sparse_dim, nnz, dtype=None):
    """
    Parameters
    ----------
    size : tuple
    sparse_dim : int
    nnz : int
    dtype : {None, dtype}

    Returns
    -------
    indices : numpy.ndarray
    values : numpy.ndarray
    """
    if dtype is None:
        dtype = 'float32'

    indices = (np.random.rand(int(nnz*1.5), sparse_dim) * np.array(size[:sparse_dim])).T.astype('int64')
    indices = np.array(sorted(set(map(tuple, indices.T))), dtype=indices.dtype)[:nnz].T
    new_nnz = indices.shape[1]
    v_size = [new_nnz] + list(size[sparse_dim:])
    values = np.random.randn(*v_size).astype(dtype)
    return indices, values

def test(size, nnz, dtype=None, repeat=1):
    if dtype is None:
        dtype = 'float32'
    # tensorflow sparse tensor supports only scalar values
    sparse_dim = len(size)

    indices, values = generate_coo_data(size, sparse_dim, nnz, dtype=dtype)
    if tf is not None:
        cpu = tf.DeviceSpec.from_string("/CPU:0")
        with tf.device(cpu):
            tf_sparse = tf.SparseTensor(indices.T, values, size)
            #tf_dense = tf.sparse.to_dense(tf_sparse)
            elapsed = 0
            d = sparse_dim - 1
            for i in range(repeat):
                start = time.time()
                tf_softmax = tf.sparse.softmax(tf_sparse)
                end = time.time()
                elapsed += (end - start)
            print(f'tensorflow.sparse.softmax(<{size}-{dtype}-tensor[nnz={nnz}]>, dim={d}) took {1e6*elapsed/repeat:.2f} us')

    if pt is not None:
        pt_sparse = pt.sparse_coo_tensor(indices, values, size, dtype=getattr(pt, dtype))
        for d in range(sparse_dim):
            elapsed = 0
            for i in range(repeat):
                start = time.time()
                pt_softmax = pt.sparse.softmax(pt_sparse, d)
                end = time.time()
                elapsed += (end - start)
            print(f'pytorch.sparse.softmax(<{size}-{dtype}-tensor[nnz={nnz}]>, dim={d}) took {1e6*elapsed/repeat:.2f} us')

            pt_dense = pt_sparse.to_dense()
            elapsed = 0
            for i in range(repeat):
                start = time.time()
                pt_softmax = pt.softmax(pt_dense, d)
                end = time.time()
                elapsed += (end - start)
            print(f'pytorch.softmax(<{size}-{dtype}-tensor>, dim={d}) took {1e6*elapsed/repeat:.2f} us')
            
for size, nnz in [
        ((100, 100), 1000),
        ((100, 100), 10000),

        ((100, 100, 100), 1000),
        ((100, 100, 100), 10000),
        ((100, 100, 100), 100000),
        #((2, 3, 5), 1000),
]:
    test(size, nnz, dtype='float64', repeat=1000)
