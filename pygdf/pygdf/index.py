# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import pandas as pd
import numpy as np
import pickle
from numba import cuda

from . import cudautils, utils, columnops
from .buffer import Buffer
from .numerical import NumericalColumn
from .column import Column
from .serialize import register_distributed_serializer


class Index(object):
    def serialize(self, serialize):
        header = {}
        header['payload'], frames = serialize(pickle.dumps(self))
        header['frame_count'] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        payload = deserialize(header['payload'],
                              frames[:header['frame_count']])
        return pickle.loads(payload)

    def take(self, indices):
        assert indices.dtype.kind in 'iu'
        if indices.size == 0:
            # Empty indices
            return RangeIndex(indices.size)
        else:
            # Gather
            index = cudautils.gather(data=self.gpu_values, index=indices)
            col = self.as_column().replace(data=Buffer(index))
            return GenericIndex(col)

    def argsort(self, ascending=True):
        return self.as_column().argsort(ascending=ascending)

    @property
    def values(self):
        return np.asarray([i for i in self.as_column()])

    def to_pandas(self):
        return pd.Index(self.as_column().to_pandas())

    @property
    def gpu_values(self):
        return self.as_column().to_gpu_array()

    def find_segments(self):
        """Return the beginning index for segments

        Returns
        -------
        result : NumericalColumn
        """
        segments, _ = self._find_segments()
        return segments

    def _find_segments(self):
        seg, markers = cudautils.find_segments(self.gpu_values)
        return NumericalColumn(data=Buffer(seg), dtype=seg.dtype), markers

    @classmethod
    def _concat(cls, objs):
        data = Column._concat([o.as_column() for o in objs])
        # TODO: add ability to concatenate indices without always casting to
        # `GenericIndex`
        return GenericIndex(data)

    def __eq__(self, other):
        if not isinstance(other, Index):
            return NotImplemented
        elif len(self) != len(other):
            return False

        lhs = self.as_column()
        rhs = other.as_column()
        res = lhs.unordered_compare('eq', rhs).all()
        return res

    def join(self, other, how='left', return_indexers=False):
        column_join_res = self.as_column().join(
            other.as_column(), how=how, return_indexers=return_indexers)
        if return_indexers:
            joined_col, indexers = column_join_res
            joined_index = GenericIndex(joined_col)
            return joined_index, indexers
        else:
            return column_join_res


class EmptyIndex(Index):
    """
    A singleton class to represent an empty index when a DataFrame is created
    without any initializer.
    """
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = object.__new__(EmptyIndex)
        return cls._singleton

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self
        raise IndexError

    def __len__(self):
        return 0

    def as_column(self):
        buf = Buffer(np.empty(0, dtype=np.int64))
        return NumericalColumn(data=buf, dtype=buf.dtype)

    def find_label_range(self, first, last):
        return None, None


class RangeIndex(Index):
    """Basic start..stop
    """
    def __init__(self, start, stop=None):
        """RangeIndex(size), RangeIndex(start, stop)

        Parameters
        ----------
        size, start, stop: int
        """
        if stop is None:
            start, stop = 0, start
        self._start = int(start)
        self._stop = int(stop)

    def __repr__(self):
        return "{}(start={}, stop={})".format(self.__class__.__name__,
                                              self._start, self._stop)

    def __len__(self):
        return self._stop - self._start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop = utils.normalize_slice(index, len(self))
            start += self._start
            stop += self._start
            if index.step is None:
                return RangeIndex(start, stop)
            else:
                return index_from_range(start, stop, index.step)
        elif isinstance(index, int):
            index = utils.normalize_index(index, len(self))
            index += self._start
            return index
        else:
            raise ValueError(index)

    def __eq__(self, other):
        if isinstance(other, EmptyIndex):
            return len(self) == 0
        elif isinstance(other, RangeIndex):
            return (self._start == other._start and self._stop == other._stop)
        else:
            return super(RangeIndex, self).__eq__(other)

    @property
    def dtype(self):
        return np.dtype(np.int64)

    def find_label_range(self, first, last):
        # clip first to range
        if first is None or first < self._start:
            begin = self._start
        elif first < self._stop:
            begin = first
        else:
            begin = self._stop
        # clip last to range
        if last is None:
            end = self._stop
        elif last < self._start:
            end = begin
        elif last < self._stop:
            end = last + 1
        else:
            end = self._stop
        # shift to index
        return begin - self._start, end - self._start

    def as_column(self):
        if len(self) > 0:
            vals = cudautils.arange(self._start, self._stop, dtype=self.dtype)
        else:
            vals = cuda.device_array(0, dtype=self.dtype)
        return NumericalColumn(data=Buffer(vals), dtype=vals.dtype)

    def to_pandas(self):
        return pd.RangeIndex(start=self._start, stop=self._stop,
                             dtype=self.dtype)


def index_from_range(start, stop=None, step=None):
    vals = cudautils.arange(start, stop, step, dtype=np.int64)
    return GenericIndex(NumericalColumn(data=Buffer(vals), dtype=vals.dtype))


class GenericIndex(Index):
    def __new__(self, values):
        from .series import Series

        # normalize the input
        if isinstance(values, Series):
            values = values._column
        elif isinstance(values, columnops.TypedColumnBase):
            values = values
        else:
            values = NumericalColumn(data=Buffer(values), dtype=values.dtype)

        assert isinstance(values, columnops.TypedColumnBase), type(values)
        assert values.null_count == 0

        # return the index instance
        if len(values) == 0:
            # for empty index, return a EmptyIndex instead
            return EmptyIndex()
        else:
            # Make GenericIndex object
            res = Index.__new__(GenericIndex)
            res._values = values
            return res

    def serialize(self, serialize):
        header = {}
        header['payload'], frames = serialize(self._values)
        header['frame_count'] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        payload = deserialize(header['payload'],
                              frames[:header['frame_count']])
        return cls(payload)

    def __sizeof__(self):
        return self._values.__sizeof__()

    def __reduce__(self):
        return GenericIndex, tuple([self._values])

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        vals = [self._values[i] for i in range(min(len(self), 10))]
        return "{}({}, dtype={})".format(self.__class__.__name__,
                                         vals, self._values.dtype)

    def __getitem__(self, index):
        res = self._values[index]
        if not isinstance(index, int):
            return GenericIndex(res)
        else:
            return res

    def as_column(self):
        """Convert the index as a Series.
        """
        return self._values

    @property
    def dtype(self):
        return self._values.dtype

    def find_label_range(self, first, last):
        """Find range that starts with *first* and ends with *last*,
        inclusively.

        Returns
        -------
        begin, end : 2-tuple of int
            The starting index and the ending index.
            The *last* value occurs at ``end - 1`` position.
        """
        col = self._values
        begin, end = None, None
        if first is not None:
            begin = col.find_first_value(first)
        if last is not None:
            end = col.find_last_value(last)
            end += 1
        return begin, end


register_distributed_serializer(RangeIndex)
register_distributed_serializer(GenericIndex)
