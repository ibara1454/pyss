#! /usr/bin/env python
# -*- coding: utf-8 -*-


def replace_attr_if_match(mapper, dic):
    """
    Parameters
    ----------
    mapper : sorted list [(key, [(pattern, value)])]
    dic : sorted list [(key, value)]

    Returns
    -------
    dic : dict
    dict mapped by mapper.

    Examples
    --------
    >>> dict_1 = {
    >>>     'key1': 'pattern1_2',
    >>>     'key2': 'pattern2_1',
    >>>     'key3': 'pattern3_3',
    >>>     'key4': 'pattern4_1'
    >>> }
    >>> mapper = {
    >>>     'key1': [('pattern1_1', 'value1'), ('pattern1_2', 'value2')],
    >>>     'key2': [('pattern2_1', 'value1')],
    >>>     'key3': [('pattern3_1', 'value1')],
    >>> }
    >>> result = replace_attr_if_match(mapper, dict_1)
    >>> result
    {'key1': 'value2', 'key2': 'value1', 'key3': 'pattern3_3',
    'key4': 'pattern4_1'}
    """
    sorted_mapper = sorted(mapper.items())
    sorted_dic = sorted(dic.items())
    res = __match_map(sorted_mapper, sorted_dic, [])
    return dict(res)


def replace_if_match(xs, y):
    """
    Find the first pattern match in list.

    Return the corresponding value of patterns if matches,
    return the same value of x if there has no match.
    Parameters
    ----------
    xs : [(pattern, value)]
    y : object

    Returns
    -------
    result : object
    """
    # Find the first match in ys
    matches = next(filter(lambda tup: tup[0] == y, xs), None)
    if matches is None:
        return y
    else:
        return matches[1]


def __match_map(mapper, xs, res):
    """
    Parameters
    ----------
    mapper : sorted list [(key, [(pattern, value)])]
    xs : sorted list [(key, value)]
    res : list of (key, value) tuples

    Returns
    -------
    res : list of (key, value) tuples
    """
    if not mapper or not xs:
        return res + xs
    else:
        m, *mtail = mapper
        x, *xtail = xs
        order = __compare_first(m, x)
        if order > 0:
            return __match_map(mapper, xtail, res + [x])
        elif order < 0:
            return __match_map(mtail, xs, res)
        else:  # order == 0
            mkey, mvalues = m  # mkey and xkey are same value
            xkey, xvalue = x
            match_value = replace_if_match(mvalues, xvalue)
            return __match_map(mtail, xtail, res + [(xkey, match_value)])


def __compare_first(xs, ys):
    """
    Compare the first element of two iteratable objects.
    Parameters
    ----------
    xs : [comparable, ...] iterable
    ys : [comparable, ...] iterable

    Returns
    -------
    result : int
        Returns 1 if x's first element is larger than y's,
        -1 if x's is smaller then y's, 0 if x's equals y's.
    """
    return __compare(next(iter(xs)), next(iter(ys)))


def __compare(x, y):
    """
    Parameters
    ----------
    x : comparable
    y : comparable

    Returns
    -------
    result : int
        Returns 1 if x is larger than y, -1 if x is smaller
        then y, 0 if x equals y.
    """
    if x > y:
        return 1
    elif x < y:
        return -1
    else:
        return 0