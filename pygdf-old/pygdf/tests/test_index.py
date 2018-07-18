"""
Test related to Index
"""
import pytest
import itertools

import numpy as np

from pygdf.dataframe import DataFrame
from pygdf.index import EmptyIndex, GenericIndex, RangeIndex


def test_df_set_index_from_series():
    df = DataFrame()
    df['a'] = list(range(10))
    df['b'] = list(range(0, 20, 2))

    # Check set_index(Series)
    df2 = df.set_index(df['b'])
    assert list(df2.columns) == ['a', 'b']
    sliced_strided = df2.loc[2:6]
    print(sliced_strided)
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_set_index_from_name():
    df = DataFrame()
    df['a'] = list(range(10))
    df['b'] = list(range(0, 20, 2))

    # Check set_index(column_name)
    df2 = df.set_index('b')
    print(df2)
    # 1 less column because 'b' is used as index
    assert list(df2.columns) == ['a']
    sliced_strided = df2.loc[2:6]
    print(sliced_strided)
    assert len(sliced_strided) == 3
    assert list(sliced_strided.index.values) == [2, 4, 6]


def test_df_slice_empty_index():
    df = DataFrame()
    assert isinstance(df.index, EmptyIndex)
    assert isinstance(df.index[:1], EmptyIndex)
    with pytest.raises(IndexError):
        df.index[1]


def test_index_find_label_range():
    idx = GenericIndex(np.asarray([4, 5, 6, 10]))
    assert idx.find_label_range(4, 6) == (0, 3)
    assert idx.find_label_range(5, 10) == (1, 4)
    # Last value not found
    with pytest.raises(ValueError) as raises:
        idx.find_label_range(0, 6)
    raises.match("value not found")
    # Last value not found
    with pytest.raises(ValueError) as raises:
        idx.find_label_range(4, 11)
    raises.match("value not found")


def test_index_comparision():
    start, stop = 10, 34
    rg = RangeIndex(start, stop)
    gi = GenericIndex(np.arange(start, stop))
    assert rg == gi
    assert gi == rg
    assert rg[:-1] != gi
    assert rg[:-1] == gi[:-1]
    assert EmptyIndex() != rg
    assert EmptyIndex() != gi
