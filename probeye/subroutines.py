# standard library imports
from copy import copy

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

def underlined_string(string, symbol="═", n_empty_start=1, n_empty_end=1):
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

def titled_table(title_str, table_str, symbol='-', n_empty_start=1,
                 n_empty_end=0):
    """
    Adds an underlined title string to a given table string. The line, that
    underlines the title will be as long as the longest line of the table.

    Parameters
    ----------
    title_str : string
        The title to be put on top of the table.
    table_str : string
        The string representing the table. For example generated using tabulate.
    symbol : string
        A single character the line should be 'made' of.
    n_empty_start : int, optional
        Number of empty lines added before the title.
    n_empty_end : int, optional
        Number of empty lines added after the table string.

    Returns
    -------
    result_string : string
        An underlined title, followed by a table.
    """
    # get the number of characters in the given table's longest line
    max_line_length = max([len(line) for line in table_str.split('\n')])
    # now, simply concatenate the different lines
    result_string = n_empty_start * '\n' +\
                    title_str + '\n' +\
                    max_line_length * symbol + '\n' +\
                    table_str + '\n' +\
                    n_empty_end * '\n'
    return result_string

def replace_string_chars(string, replace=None, remove=None):
    """
    Removes and replaces characters from a given string according to the input.

    Parameters
    ----------
    string : string
        The string to be modified.
    replace : dict, optional
        The keys are the characters to be replaced, the values are stating their
        replacements.
    remove : list, optional
        A list of characters to be removed from the given string.

    Returns
    -------
     string : string
        The modified string with removed/replaced characters.
    """
    # first, do the replacements
    if replace is not None:
        for char, replacement in replace.items():
            string = string.replace(char, replacement)
    # finally, remove characters as requested
    if remove is not None:
        for char in remove:
            string = string.replace(char, '')
    return string

def simplified_list_string(list_):
    """
    Given a list, it returns the string returned by its __str__ method, where
    some characters are removed for a slightly easier to read printout. Example:
    the list ['a', 1.2] is usually printed as '['a', 1.2]'. Here, it will be
    converted into the string 'a, 1.2' (no brackets, and no quotes).

    Parameters
    ----------
    list_ : list
        Some list to be printed.

    Returns
    -------
    simplified_list_str : string
        The list_'s __str__ method's return string without brackets and quotes.
    """
    simplified_list_str = replace_string_chars(
        str(list_), remove=['[', ']', "'"])
    return simplified_list_str

def simplified_dict_string(dict_):
    """
    Given a dictionary, it returns the string returned by its __str__ method,
    where some characters are removed for a slightly easier to read printout.
    Example: the dict {'a': 1.2} is usually printed as '{'a': 1.2, 'b': 2.1}'.
    Here, it will be converted into the string 'a=1.2, b=2.1'.

    Parameters
    ----------
    dict_ : dict
        Some dictionary to be printed.

    Returns
    -------
    simplified_dict_str : string
        Modified version of dict_'s __str__ method's return string (no quotes,
        no braces and the colon will be replaced with an equal sign.
    """
    simplified_dict_str = replace_string_chars(
        str(dict_), remove=['{', '}', "'"], replace={': ': '='})
    return simplified_dict_str

def unvectorize_dict_values(dict_):
    """
    Takes a dict with items like <name>: <vector> and converts it into a list,
    where each element is a 'fraction' or the whole dictionary. The following
    example will illustrate it: {'x': [1, 2, 3], 'y': [4, 5, 6]} will be
    converted into [{'x': 1, 'y': 4}, {'x': 2, 'y': 5}, {'x': 3, 'y': 6}].

    Parameters
    ----------
    dict_ : dict
        The dictionary that should be converted. All values must be 1D arrays
        of the same length.

    Returns
    -------
    result_list : list
        The 'un-vectorized' dictionary. Check out the example above.
    """

    # all values must be iterable
    dict_copy = copy(dict_)
    for key, value in dict_.items():
        if not hasattr(value, '__len__'):
            dict_copy[key] = [value]

    # check if all lengths are the same
    if len({len(vector) for vector in dict_copy.values()}) != 1:
        raise RuntimeError(
            "The values of the dictionary have different lengths!"
        )

    # create the result list
    vector_length = len([*dict_copy.values()][0])
    keys = [*dict_.keys()]
    result_list = []
    for i in range(vector_length):
        atom_dict = dict()
        for key in keys:
            atom_dict[key] = dict_copy[key][i]
        result_list.append(atom_dict)

    return result_list

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
