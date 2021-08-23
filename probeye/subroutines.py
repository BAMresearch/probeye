def len_or_one(obj):
    """
    Returns the length of an object or 1 if no length is defined.

    Parameters
    ----------
    obj : obj
        Most of the time this will be a list/tuple or a single scalar number.

    Returns
    -------
    length : int
        The length of the given list/tuple etc. or 1, if obj has no __len__
        attribute; the latter case is mostly intended for scalar numbers

    """
    if hasattr(obj, '__len__'):
        length = len(obj)
    else:
        length = 1
    return length


def delta_x(x0, delta=None):
    """
    Returns a good choice for the step size for numeric differentiation around
    x0 when using a simple two-point approximation, see also the value h in
    https://en.wikipedia.org/wiki/Numerical_differentiation.

    Parameters
    ----------
    x0 : float
        Point where the derivative should be evaluated.
    delta : float, optional
        This parameter can be used when a specific fixed value should be
        returned; might be used for debugging.

    Returns
    -------
    dx : float
        Computed step size for numeric differentiation.
    """
    # use the delta-value if delta is specified
    if delta is not None:
        return delta
    eps = 1e-7  # approx sqrt(machine precision)
    dx = x0 * eps + eps  # TODO: clarify if the '+ eps' part is correct
    if dx == 0:
        dx = eps
    return dx

def underlined_string(string, symbol="=", n_empty_start=1, n_empty_end=1):
    """
    Adds a line made of 'symbol'-characters under a given string and returns it.

    Parameters
    ----------
    string : string
        The string that should be underlined
    symbol : string
        A single character the line should be 'made' of
    n_empty_start : int, optional
        Number of empty lines added before the underlined string
    n_empty_end : int, optional
        Number of empty lines added after the underlined string

    Returns
    -------
    result_string : string
        The generated string representing an underlined string, possibly with
        empty lines added before/after
    """
    n_chars = len(string)
    underline_string = n_chars * symbol
    empty_lines_start = n_empty_start * "\n"
    empty_lines_end = n_empty_end * "\n"
    result_string = string + "\n" + underline_string
    result_string = empty_lines_start + result_string + empty_lines_end
    return result_string

def sub_when_empty(string, empty_str="-"):
    """
    Just returns a given string if it is not empty. If it is empty though, a
    default string is returned instead.

    Parameters
    ----------
    string : string
        The string to check if it is empty or not
    empty_str : string, optional
        The string to be returned if the given string is empty

    Returns
    -------
    result_string : string
        Either the given string (when 'string' is not empty) or the empty_str
        (when 'string' is empty)
    """
    if len(string) > 0:
        result_string = string
    else:
        result_string = empty_str
    return result_string

def tcs(string_1, string_2, sep=":", col_width=24, empty_str="-"):
    """
    Returns a two-column-string (tcs) made from two given strings with a
    separator in between. The two features are the definable with of the first
    column and the replacement of empty strings in the second column. The
    created output looks like 'alpha:    1.45'.

    Parameters
    ----------
    string_1 : string
        The string to appear in the left column
    string_2 : string, list
        The string to appear in the right column. If a list is given, the string
        returned by the list's __str__() method will be printed.
    sep : string, optional
        A single character to appear right after string_1
    col_width : int, optional
        The column width of the first column
    empty_str : string, optional
        The replacement string of string_2 if the latter is empty

    Returns
    -------
    result_string : string
        The created string representing a two-column-string
    """
    first_column = string_1+sep
    second_column = sub_when_empty(string_2, empty_str=empty_str)
    result_string = f"{first_column:{col_width}s}{second_column}\n"
    return result_string

def list2dict(list_):
    """
    Converts a list into a specific dictionary. The list may only contain
    strings or one-element dictionaries. For example [{'a': 'm'}, 'b'] will be
    converted into {'a': 'm', 'b': 'b'}.

    Parameters
    ----------
    list_ : list
        May only contain strings or one-element dictionaries.

    Returns
    -------
    dict_ : dict
        Strings are mapped to themselves, while one-element dictionaries are
        simply added to this result dictionary.
    """
    if type(list_) != list:
        raise TypeError(
            f"Input argument must be of type 'list', found '{type(list_)}'."
        )
    dict_ = {}
    for element in list_:
        element_type = type(element)
        if element_type == dict:
            if len(element) != 1:
                raise ValueError(
                    f"Found a dict-element, which has {len(element)} instead "
                    f"of 1 key-value pair."
                )
            dict_ = {**dict_, **element}
        elif element_type == str:
            dict_[element] = element
        else:
            raise TypeError(
                f"The elements in the given list must be either of type "
                f"'string' or 'dict'. Found '{element_type}' however."
            )
    return dict_
