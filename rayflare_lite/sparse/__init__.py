import numpy as np


class COO:
    def __init__(
        self,
        coords,
        data=None,
        shape=None,
        has_duplicates=False,
        sorted=False,
        prune=False,
        fill_value=0,
    ):
        _ = (has_duplicates, sorted, prune)
        if isinstance(coords, COO) and data is None and shape is None:
            self.coords = coords.coords.copy()
            self.data = coords.data.copy()
            self.shape = coords.shape
            self.fill_value = coords.fill_value
            return

        if data is None and shape is None:
            dense = np.asarray(coords)
            tmp = COO.from_numpy(dense)
            self.coords = tmp.coords
            self.data = tmp.data
            self.shape = tmp.shape
            self.fill_value = tmp.fill_value
            return

        if data is None:
            raise ValueError("data must be provided when coords are explicit")

        if isinstance(coords, (list, tuple)) and not isinstance(coords, np.ndarray):
            coords_arr = np.vstack([np.asarray(c, dtype=np.intp) for c in coords])
        else:
            coords_arr = np.asarray(coords, dtype=np.intp)
            if coords_arr.ndim == 1:
                coords_arr = coords_arr.reshape(1, -1)

        self.coords = coords_arr
        self.data = np.asarray(data)
        if shape is None:
            if self.coords.size == 0:
                self.shape = (0,)
            else:
                self.shape = tuple(np.max(self.coords, axis=1) + 1)
        else:
            self.shape = tuple(shape)
        self.fill_value = fill_value

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.data.dtype

    def todense(self):
        dense = np.full(self.shape, self.fill_value, dtype=self.data.dtype)
        if self.coords.size:
            dense[tuple(self.coords)] = self.data
        return dense

    def __array__(self, dtype=None):
        dense = self.todense()
        if dtype is not None:
            return dense.astype(dtype, copy=False)
        return dense

    def __getitem__(self, key):
        return COO.from_numpy(self.todense()[key])

    def transpose(self, axes=None):
        return COO.from_numpy(self.todense().transpose(axes))

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape):
        return COO.from_numpy(self.todense().reshape(shape))

    @classmethod
    def from_numpy(cls, arr):
        dense = np.asarray(arr)
        if dense.size == 0:
            coords = np.empty((dense.ndim, 0), dtype=np.intp)
            data = dense.reshape(-1)
            return cls(coords, data=data, shape=dense.shape)
        coords = np.array(np.nonzero(dense))
        data = dense[tuple(coords)]
        return cls(coords, data=data, shape=dense.shape)

    def __repr__(self):
        return f"<COO shape={self.shape} dtype={self.data.dtype} nnz={self.data.size}>"


def stack(arrays, axis=0):
    dense = [a.todense() if isinstance(a, COO) else np.asarray(a) for a in arrays]
    return COO.from_numpy(np.stack(dense, axis=axis))


def einsum(subscripts, *operands, **kwargs):
    dense_ops = [op.todense() if isinstance(op, COO) else op for op in operands]
    result = np.einsum(subscripts, *dense_ops, **kwargs)
    if isinstance(result, np.ndarray):
        return COO.from_numpy(result)
    return result


def dot(a, b):
    a_dense = a.todense() if isinstance(a, COO) else a
    b_dense = b.todense() if isinstance(b, COO) else b
    result = np.dot(a_dense, b_dense)
    if isinstance(result, np.ndarray):
        return COO.from_numpy(result)
    return result


def save_npz(filename, matrix, compressed=True):
    nodes = {
        "data": matrix.data,
        "shape": matrix.shape,
        "fill_value": matrix.fill_value,
        "coords": matrix.coords,
    }
    if compressed:
        np.savez_compressed(filename, **nodes)
    else:
        np.savez(filename, **nodes)


def load_npz(filename):
    with np.load(filename) as fp:
        coords = fp["coords"]
        data = fp["data"]
        shape = tuple(fp["shape"])
        fill_value = fp["fill_value"][()]
        return COO(
            coords=coords,
            data=data,
            shape=shape,
            sorted=True,
            has_duplicates=False,
            fill_value=fill_value,
        )


__all__ = ["COO", "dot", "einsum", "load_npz", "save_npz", "stack"]
