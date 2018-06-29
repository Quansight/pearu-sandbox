"""
Helper functions for parameterized docstring
"""
import functools
import string
import re

_regex_whitespaces = re.compile('^\s+$')


def _only_spaces(s):
    return bool(_regex_whitespaces.match(s))


_wrapopts = {
    'width': 78,
    'replace_whitespace': False,
}


def docfmt(**kwargs):
    """Format docstring.

    Simliar to saving the result of ``__doc__.format(**kwargs)`` as the
    function's docstring.
    """
    kwargs = {k: v.lstrip() for k, v in kwargs.items()}

    def outer(fn):
        buf = []
        formatsiter = string.Formatter().parse(fn.__doc__)
        for literal, field, fmtspec, conv in formatsiter:
            assert conv is None
            assert not fmtspec
            buf.append(literal)
            if field is not None:
                # get indentation
                lines = literal.rsplit('\n', 1)
                if _only_spaces(lines[-1]):
                    indent = ' ' * len(lines[-1])
                    valuelines = kwargs[field].splitlines(True)
                    # first line
                    buf.append(valuelines[0])
                    # subsequent lines are indented
                    buf.extend([indent + ln for ln in valuelines[1:]])
                else:
                    buf.append(kwargs[field])

        fn.__doc__ = ''.join(buf)
        return fn
    return outer


def docfmt_partial(**kwargs):
    return functools.partial(docfmt, **kwargs)

