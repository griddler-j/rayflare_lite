import numpy as np


class Coord:
    def __init__(self, data, dims=None):
        self.data = np.asarray(data)
        self.dims = list(dims) if dims is not None else []


def _normalize_dims(dims, ndim):
    if dims is None:
        return [f"dim_{i}" for i in range(ndim)]
    if isinstance(dims, str):
        dims = [dims]
    dims = list(dims)
    if len(dims) != ndim:
        raise ValueError("dims length does not match data ndim")
    return dims


def _coord_from_value(key, value, dims):
    if isinstance(value, Coord):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        c_dims, c_data = value
        if isinstance(c_dims, str):
            c_dims = [c_dims]
        return Coord(c_data, c_dims)
    if key in dims:
        return Coord(value, [key])
    return Coord(value, [])


def _copy_coords(coords):
    return {k: Coord(v.data, v.dims) for k, v in coords.items()}


class DataArray:
    __array_priority__ = 1000

    def __init__(self, data, dims=None, coords=None, name=None, attrs=None):
        self._data = np.asarray(data)
        self.dims = _normalize_dims(dims, self._data.ndim)
        self.name = name
        self.attrs = attrs or {}

        coords = coords or {}
        self.coords = {}
        for key, value in coords.items():
            self.coords[key] = _coord_from_value(key, value, self.dims)

        for axis, dim in enumerate(self.dims):
            if dim not in self.coords:
                self.coords[dim] = Coord(np.arange(self._data.shape[axis]), [dim])

    @property
    def data(self):
        return self._data

    @property
    def values(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(self, key):
        if isinstance(key, dict):
            return self._sel(key)
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < self.ndim:
            key = key + (slice(None),) * (self.ndim - len(key))
        data = self._data[key]
        dims = []
        coords = _copy_coords(self.coords)
        for axis, dim in enumerate(self.dims):
            indexer = key[axis]
            if isinstance(indexer, (int, np.integer)):
                continue
            dims.append(dim)
            if dim in coords:
                coord_data = coords[dim].data
                coords[dim] = Coord(coord_data[indexer], [dim])
        coords = {k: v for k, v in coords.items() if all(d in dims for d in v.dims)}
        return DataArray(data, dims=dims, coords=coords, name=self.name)

    def _binary_op(self, other, op):
        if isinstance(other, DataArray):
            data = op(self._data, other._data)
            coords = _copy_coords(self.coords)
            dims = list(self.dims)
        else:
            data = op(self._data, other)
            coords = _copy_coords(self.coords)
            dims = list(self.dims)
        return DataArray(data, dims=dims, coords=coords, name=self.name)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        return DataArray(np.subtract(other, self._data), dims=self.dims, coords=_copy_coords(self.coords))

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other):
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __rtruediv__(self, other):
        return DataArray(np.divide(other, self._data), dims=self.dims, coords=_copy_coords(self.coords))

    def __gt__(self, other):
        return self._binary_op(other, np.greater)

    def __ge__(self, other):
        return self._binary_op(other, np.greater_equal)

    def __lt__(self, other):
        return self._binary_op(other, np.less)

    def __le__(self, other):
        return self._binary_op(other, np.less_equal)

    def __eq__(self, other):
        return self._binary_op(other, np.equal)

    def where(self, cond, other=np.nan):
        cond_data = cond.data if isinstance(cond, DataArray) else cond
        other_data = other.data if isinstance(other, DataArray) else other
        data = np.where(cond_data, self._data, other_data)
        return DataArray(data, dims=self.dims, coords=_copy_coords(self.coords), name=self.name)

    def fillna(self, value):
        data = np.where(np.isnan(self._data), value, self._data)
        return DataArray(data, dims=self.dims, coords=_copy_coords(self.coords), name=self.name)

    def rename(self, name_or_dict):
        if isinstance(name_or_dict, dict):
            mapping = name_or_dict
            dims = [mapping.get(dim, dim) for dim in self.dims]
            coords = {}
            for key, coord in self.coords.items():
                new_key = mapping.get(key, key)
                new_dims = [mapping.get(d, d) for d in coord.dims]
                coords[new_key] = Coord(coord.data, new_dims)
            return DataArray(self._data, dims=dims, coords=coords, name=self.name)
        return DataArray(self._data, dims=self.dims, coords=_copy_coords(self.coords), name=name_or_dict)

    def assign_coords(self, coords=None, **kwargs):
        new = DataArray(self._data, dims=self.dims, coords=_copy_coords(self.coords), name=self.name)
        update = coords or {}
        update.update(kwargs)
        for key, value in update.items():
            new.coords[key] = _coord_from_value(key, value, new.dims)
        return new

    def expand_dims(self, dim):
        if isinstance(dim, str):
            dim = [dim]
        dims = list(dim) + list(self.dims)
        data = self._data
        for _ in dim:
            data = np.expand_dims(data, axis=0)
        coords = _copy_coords(self.coords)
        for new_dim in dim:
            if new_dim in coords:
                coord_data = np.asarray(coords[new_dim].data)
                if coord_data.ndim == 0:
                    coord_data = np.array([coord_data.item()])
                coords[new_dim] = Coord(coord_data, [new_dim])
            else:
                coords[new_dim] = Coord(np.array([0]), [new_dim])
        return DataArray(data, dims=dims, coords=coords, name=self.name)

    def transpose(self, *dims):
        if not dims:
            dims = tuple(reversed(self.dims))
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        axes = [self.dims.index(dim) for dim in dims]
        data = np.transpose(self._data, axes=axes)
        coords = _copy_coords(self.coords)
        return DataArray(data, dims=list(dims), coords=coords, name=self.name)

    def reduce(self, func, dim=None, axis=None, **kwargs):
        if axis is None:
            if dim is None:
                axis = None
            else:
                if isinstance(dim, str):
                    dim = [dim]
                axis = tuple(self.dims.index(d) for d in dim)
        data = func(self._data, axis=axis, **kwargs)
        if axis is None:
            dims = []
        else:
            axes = set(axis if isinstance(axis, tuple) else (axis,))
            dims = [d for i, d in enumerate(self.dims) if i not in axes]
        coords = {k: v for k, v in self.coords.items() if all(d in dims for d in v.dims)}
        return DataArray(data, dims=dims, coords=coords, name=self.name)

    def interp(self, **coords):
        result = self
        for dim, new_x in coords.items():
            if dim not in result.dims:
                continue
            axis = result.dims.index(dim)
            old_x = result.coords[dim].data
            new_x = np.asarray(new_x)

            def _interp_1d(y):
                return np.interp(new_x, old_x, y)

            data = np.apply_along_axis(_interp_1d, axis, result._data)
            new_coords = _copy_coords(result.coords)
            new_coords[dim] = Coord(new_x, [dim])
            result = DataArray(data, dims=result.dims, coords=new_coords, name=result.name)
        return result

    def groupby(self, group):
        return _GroupBy(self, group)

    def _slice_by_indices(self, axis, indices):
        indexer = [slice(None)] * self.ndim
        indexer[axis] = indices
        data = self._data[tuple(indexer)]
        dims = list(self.dims)
        coords = _copy_coords(self.coords)
        if isinstance(indices, (int, np.integer)):
            dims.pop(axis)
            coords = {k: v for k, v in coords.items() if all(d in dims for d in v.dims)}
        else:
            dim = dims[axis]
            if dim in coords:
                coords[dim] = Coord(coords[dim].data[indices], [dim])
            for key, coord in list(coords.items()):
                if dim in coord.dims and coord.dims != [dim]:
                    coords[key] = Coord(coord.data, coord.dims)
        return DataArray(data, dims=dims, coords=coords, name=self.name)

    def _label_to_index(self, coord_data, value, method=None):
        arr = np.asarray(coord_data)
        if method == "nearest":
            return int(np.abs(arr - value).argmin())
        matches = np.where(arr == value)[0]
        if matches.size == 0:
            raise KeyError(f"label {value} not found")
        return int(matches[0])

    def _sel(self, indexers, method=None):
        data = self._data
        dims = list(self.dims)
        coords = _copy_coords(self.coords)
        drop_dims = set()
        indexer_list = [slice(None)] * self.ndim
        for key, value in indexers.items():
            if key in self.dims:
                dim = key
            elif key in self.coords and self.coords[key].dims:
                dim = self.coords[key].dims[0]
            else:
                continue
            axis = self.dims.index(dim)
            if isinstance(value, (list, tuple, np.ndarray)):
                indices = [self._label_to_index(self.coords[dim].data, v, method=method) for v in value]
                indexer_list[axis] = indices
            else:
                idx = self._label_to_index(self.coords[dim].data, value, method=method)
                indexer_list[axis] = idx
                drop_dims.add(dim)
        data = data[tuple(indexer_list)]
        if drop_dims:
            dims = [d for d in dims if d not in drop_dims]
            coords = {k: v for k, v in coords.items() if all(d in dims for d in v.dims)}
        return DataArray(data, dims=dims, coords=coords, name=self.name)

    def _set_sel(self, indexers, value):
        indexer_list = [slice(None)] * self.ndim
        for key, value_key in indexers.items():
            if key in self.dims:
                dim = key
            elif key in self.coords and self.coords[key].dims:
                dim = self.coords[key].dims[0]
            else:
                continue
            axis = self.dims.index(dim)
            idx = self._label_to_index(self.coords[dim].data, value_key)
            indexer_list[axis] = idx
        self._data[tuple(indexer_list)] = value

    @property
    def loc(self):
        return _LocIndexer(self)

    def sel(self, method=None, **indexers):
        return self._sel(indexers, method=method)

    @property
    def plot(self):
        return _PlotAccessor(self)


class _PlotAccessor:
    def __init__(self, data_array):
        self._da = data_array

    def imshow(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        return ax.imshow(self._da.data, **kwargs)


class _LocIndexer:
    def __init__(self, data_array):
        self._da = data_array

    def __getitem__(self, indexers):
        return self._da._sel(indexers)

    def __setitem__(self, indexers, value):
        self._da._set_sel(indexers, value)


class _GroupBy:
    def __init__(self, data_array, group):
        self._da = data_array
        self._group = group
        self._flat = False
        if group in data_array.dims:
            self._dim = group
            self._coord = data_array.coords[group]
        elif group in data_array.coords and data_array.coords[group].dims:
            self._dim = data_array.coords[group].dims[0]
            self._coord = data_array.coords[group]
        else:
            raise KeyError(f"group {group} not found")
        if self._coord.dims != [self._dim]:
            self._flat = True

    def map(self, func, args=None, **kwargs):
        args = args or ()
        values = np.asarray(self._coord.data)
        if self._flat:
            flat_values = values.ravel()
            seen = set()
            groups = []
            for v in flat_values:
                if v not in seen:
                    seen.add(v)
                    groups.append(v)

            out = np.empty_like(self._da.data, dtype=self._da.data.dtype)
            for g in groups:
                mask = values == g
                subset_data = self._da.data[mask]
                subset = DataArray(
                    subset_data,
                    dims=["stacked"],
                    coords={self._group: Coord(np.full(subset_data.shape, g), ["stacked"])},
                    name=self._da.name,
                )
                res = func(subset, *args, **kwargs)
                res_data = res.data if isinstance(res, DataArray) else np.asarray(res)
                out[mask] = res_data
            return DataArray(out, dims=self._da.dims, coords=_copy_coords(self._da.coords), name=self._da.name)

        axis = self._da.dims.index(self._dim)
        seen = set()
        groups = []
        for v in values:
            if v not in seen:
                seen.add(v)
                groups.append(v)

        results = []
        result_same_shape = None
        for g in groups:
            indices = np.where(values == g)[0]
            subset = self._da._slice_by_indices(axis, indices)
            res = func(subset, *args, **kwargs)
            if isinstance(res, DataArray):
                res_data = res.data
            else:
                res_data = np.asarray(res)
            if result_same_shape is None:
                result_same_shape = res_data.ndim == self._da.ndim
            results.append(res_data)

        if result_same_shape:
            out = np.empty_like(self._da.data, dtype=np.asarray(results[0]).dtype)
            for g, res_data in zip(groups, results):
                indices = np.where(values == g)[0]
                indexer = [slice(None)] * self._da.ndim
                indexer[axis] = indices
                out[tuple(indexer)] = res_data
            return DataArray(out, dims=self._da.dims, coords=_copy_coords(self._da.coords), name=self._da.name)

        out = np.stack(results, axis=axis)
        dims = list(self._da.dims)
        dims[axis] = self._group
        coords = {k: v for k, v in self._da.coords.items() if self._dim not in v.dims}
        coords[self._group] = Coord(np.array(groups), [self._group])
        return DataArray(out, dims=dims, coords=coords, name=self._da.name)


class Dataset:
    def __init__(self, data_vars, coords=None):
        self.data_vars = dict(data_vars)
        if coords is None:
            coords = {}
            for da in self.data_vars.values():
                for key, coord in da.coords.items():
                    if key not in coords:
                        coords[key] = Coord(coord.data, coord.dims)
        self.coords = coords

    def __getitem__(self, key):
        return self.data_vars[key]

    def merge(self, other):
        return merge([self, other])

    def reduce(self, func, dim=None, axis=None, **kwargs):
        data_vars = {
            name: da.reduce(func, dim=dim, axis=axis, **kwargs)
            for name, da in self.data_vars.items()
        }
        coords = {}
        for da in data_vars.values():
            for key, coord in da.coords.items():
                if key not in coords:
                    coords[key] = Coord(coord.data, coord.dims)
        return Dataset(data_vars, coords=coords)

    def assign_coords(self, coords=None, **kwargs):
        update = coords or {}
        update.update(kwargs)
        new_coords = _copy_coords(self.coords)
        for key, value in update.items():
            new_coords[key] = _coord_from_value(key, value, [])
        data_vars = {}
        for name, da in self.data_vars.items():
            data_vars[name] = da.assign_coords(update)
        return Dataset(data_vars, coords=new_coords)

    def expand_dims(self, dim):
        data_vars = {name: da.expand_dims(dim) for name, da in self.data_vars.items()}
        coords = {}
        for da in data_vars.values():
            for key, coord in da.coords.items():
                if key not in coords:
                    coords[key] = Coord(coord.data, coord.dims)
        return Dataset(data_vars, coords=coords)

    def _sel(self, indexers, method=None):
        data_vars = {}
        for name, da in self.data_vars.items():
            data_vars[name] = da._sel(indexers, method=method)
        coords = {}
        for da in data_vars.values():
            for key, coord in da.coords.items():
                if key not in coords:
                    coords[key] = Coord(coord.data, coord.dims)
        return Dataset(data_vars, coords=coords)

    @property
    def loc(self):
        return _DatasetLocIndexer(self)

    def sel(self, method=None, **indexers):
        return self._sel(indexers, method=method)

    def to_netcdf(self, *_args, **_kwargs):
        raise NotImplementedError("to_netcdf is not supported in the lite xarray shim")


class _DatasetLocIndexer:
    def __init__(self, dataset):
        self._ds = dataset

    def __getitem__(self, indexers):
        return self._ds._sel(indexers)


def concat(objs, dim="concat_dim"):
    if not objs:
        raise ValueError("concat requires at least one object")
    first = objs[0]
    if not isinstance(first, DataArray):
        raise NotImplementedError("concat only supports DataArray in the lite shim")
    if dim in first.dims:
        axis = first.dims.index(dim)
        data = np.concatenate([obj.data for obj in objs], axis=axis)
        coords = _copy_coords(first.coords)
        if dim in coords:
            coords[dim] = Coord(
                np.concatenate([obj.coords[dim].data for obj in objs]), [dim]
            )
        return DataArray(data, dims=first.dims, coords=coords, name=first.name)

    data = np.stack([obj.data for obj in objs], axis=0)
    dims = [dim] + list(first.dims)
    coords = _copy_coords(first.coords)
    coords[dim] = Coord(np.arange(len(objs)), [dim])
    return DataArray(data, dims=dims, coords=coords, name=first.name)


def merge(objs):
    data_vars = {}
    coords = {}
    for obj in objs:
        if isinstance(obj, DataArray):
            if not obj.name:
                raise ValueError("DataArray must have a name to merge")
            data_vars[obj.name] = obj
        elif isinstance(obj, Dataset):
            data_vars.update(obj.data_vars)
        else:
            raise TypeError("merge expects DataArray or Dataset")
    for da in data_vars.values():
        for key, coord in da.coords.items():
            if key not in coords:
                coords[key] = Coord(coord.data, coord.dims)
    return Dataset(data_vars, coords=coords)


def where(condition, x, y):
    cond_data = condition.data if isinstance(condition, DataArray) else condition
    x_data = x.data if isinstance(x, DataArray) else x
    y_data = y.data if isinstance(y, DataArray) else y
    data = np.where(cond_data, x_data, y_data)
    if isinstance(x, DataArray):
        base = x
    elif isinstance(y, DataArray):
        base = y
    else:
        return data
    return DataArray(data, dims=base.dims, coords=_copy_coords(base.coords), name=base.name)
