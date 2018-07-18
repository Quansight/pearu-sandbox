import pytest

import numpy as np
import pandas as pd

from pygdf.dataframe import Series, DataFrame


def test_categorical_basic():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series(cat)
    np.testing.assert_array_equal(cat.codes, sr.to_array())
    assert sr.dtype == pdsr.dtype

    # Test attributes
    assert tuple(pdsr.cat.categories) == tuple(sr.cat.categories)
    assert pdsr.cat.ordered == sr.cat.ordered

    np.testing.assert_array_equal(pdsr.cat.codes.data, sr.cat.codes.to_array())
    np.testing.assert_array_equal(pdsr.cat.codes.dtype, sr.cat.codes.dtype)

    string = str(sr)
    expect_str = """
0 a
1 a
2 b
3 c
4 a
"""
    assert all(x == y for x, y in zip(string.split(), expect_str.split()))


def test_categorical_integer():
    cat = pd.Categorical(['a', '_', '_', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series(cat)
    np.testing.assert_array_equal(cat.codes, sr.to_array(fillna='pandas'))
    assert sr.null_count == 2

    np.testing.assert_array_equal(pdsr.cat.codes.data,
                                  sr.cat.codes.fillna(-1).to_array())
    np.testing.assert_equal(pdsr.cat.codes.dtype, sr.cat.codes.dtype)

    string = str(sr)
    expect_str = """
0 a
1
2
3 c
4 a
"""
    assert string.split() == expect_str.split()


def test_categorical_compare_unordered():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)

    sr = Series(cat)

    # test equal
    out = sr == sr
    assert out.dtype == np.bool_
    assert type(out[0]) == np.bool_
    assert np.all(out)
    assert np.all(pdsr == pdsr)

    # test inequal
    out = sr != sr
    assert not np.any(out)
    assert not np.any(pdsr != pdsr)

    assert not pdsr.cat.ordered
    assert not sr.cat.ordered

    # test using ordered operators
    with pytest.raises(TypeError) as raises:
        pdsr < pdsr

    raises.match("Unordered Categoricals can only compare equality or not")

    with pytest.raises(TypeError) as raises:
        sr < sr

    raises.match("Unordered Categoricals can only compare equality or not")


def test_categorical_compare_ordered():
    cat1 = pd.Categorical(['a', 'a', 'b', 'c', 'a'],
                          categories=['a', 'b', 'c'], ordered=True)
    pdsr1 = pd.Series(cat1)
    sr1 = Series(cat1)
    cat2 = pd.Categorical(['a', 'b', 'a', 'c', 'b'],
                          categories=['a', 'b', 'c'], ordered=True)
    pdsr2 = pd.Series(cat2)
    sr2 = Series(cat2)

    # test equal
    out = sr1 == sr1
    assert out.dtype == np.bool_
    assert type(out[0]) == np.bool_
    assert np.all(out)
    assert np.all(pdsr1 == pdsr1)

    # test inequal
    out = sr1 != sr1
    assert not np.any(out)
    assert not np.any(pdsr1 != pdsr1)

    assert pdsr1.cat.ordered
    assert sr1.cat.ordered

    # test using ordered operators
    np.testing.assert_array_equal(pdsr1 < pdsr2, sr1 < sr2)
    np.testing.assert_array_equal(pdsr1 > pdsr2, sr1 > sr2)


def test_categorical_binary_add():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series(cat)

    with pytest.raises(TypeError) as raises:
        pdsr + pdsr
    raises.match('Categorical cannot perform the operation \+')

    with pytest.raises(TypeError) as raises:
        sr + sr
    raises.match('Categorical cannot perform the operation: add')


def test_categorical_unary_ceil():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series(cat)

    with pytest.raises(AttributeError) as raises:
        pdsr.ceil()
    raises.match(r'''no attribute ['"]ceil['"]''')

    with pytest.raises(TypeError) as raises:
        sr.ceil()
    raises.match('Categorical cannot perform the operation: ceil')


def test_categorical_element_indexing():
    """
    Element indexing to a cat column must give the underlying object
    not the numerical index.
    """
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series(cat)
    assert list(pdsr) == list(sr)
    assert list(pdsr.cat.codes) == list(sr.cat.codes)


def test_categorical_masking():
    """
    Test common operation for getting a all rows that matches a certain
    category.
    """
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series(cat)

    # check scalar comparison
    expect_matches = (pdsr == 'a')
    got_matches = (sr == 'a')

    print('---expect_matches---')
    print(expect_matches)
    print('---got_matches---')
    print(got_matches)
    np.testing.assert_array_equal(expect_matches.values,
                                  got_matches.to_array())

    # mask series
    expect_masked = pdsr[expect_matches]
    got_masked = sr[got_matches]

    print('---expect_masked---')
    print(expect_masked)
    print('---got_masked---')
    print(got_masked)

    assert len(expect_masked) == len(got_masked)
    assert len(expect_masked) == got_masked.valid_count
    assert list(expect_masked) == list(got_masked)


def test_df_cat_set_index():
    df = DataFrame()
    df['a'] = pd.Categorical(list('aababcabbc'), categories=list('abc'))
    df['b'] = np.arange(len(df))
    got = df.set_index('a')

    pddf = df.to_pandas()
    expect = pddf.set_index('a')

    assert list(expect.columns) == list(got.columns)
    assert list(expect.index.values) == list(got.index.values)
    np.testing.assert_array_equal(expect.index.values, got.index.values)
    np.testing.assert_array_equal(expect['b'].values, got['b'].to_array())


def test_df_cat_sort_index():
    df = DataFrame()
    df['a'] = pd.Categorical(list('aababcabbc'), categories=list('abc'))
    df['b'] = np.arange(len(df))

    got = df.set_index('a').sort_index()
    expect = df.to_pandas().set_index('a').sort_index()

    assert list(expect.columns) == list(got.columns)
    assert list(expect.index.values) == list(got.index.values)
    np.testing.assert_array_equal(expect.index.values, got.index.values)
    np.testing.assert_array_equal(expect['b'].values, got['b'].to_array())


def test_cat_series_binop_error():
    df = DataFrame()
    df['a'] = pd.Categorical(list('aababcabbc'), categories=list('abc'))
    df['b'] = np.arange(len(df))

    dfa = df['a']
    dfb = df['b']

    # lhs is a categorical
    with pytest.raises(TypeError) as raises:
        dfa + dfb
    raises.match("Categorical cannot perform the operation: add")
    # if lhs is a numerical
    with pytest.raises(TypeError) as raises:
        dfb + dfa
    raises.match("'add' operator not supported")
