# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# standard library imports
from copy import copy
from typing import Iterable

# third party imports
import numpy as np

# ============================================================================ #
#                                 Subroutines                                  #
# ============================================================================ #

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

def make_list(arg):
    """
    Converts a given argument into a list, if it is not a list or tuple. The
    typical use case for this method is to convert a single string into a list
    with this string as its only element, e.g. make_list('sigma') = ['sigma'].

    Parameters
    ----------
    arg : obj
        Essentially anything. Most of the time this might be a string or a list
        of strings or a tuple of strings.

    Returns
    -------
    new_arg : list[obj]
        Either arg if it is of type list or tuple, or a list with arg as its
        only element if arg is not of type list or tuple.
    """
    if type(arg) in [list, tuple]:
        new_arg = arg
    else:
        new_arg = [copy(arg)]
    return new_arg

def underlined_string(string, symbol="═", n_empty_start=1, n_empty_end=1):
    """
    Adds a line made of 'symbol'-characters under a given string and returns it.

    Parameters
    ----------
    string : string
        The string that should be underlined.
    symbol : string
        A single character the line should be 'made' of.
    n_empty_start : int, optional
        Number of empty lines added before the underlined string.
    n_empty_end : int, optional
        Number of empty lines added after the underlined string.

    Returns
    -------
    result_string : string
        The generated string representing an underlined string, possibly with
        empty lines added before/after.
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
    if type(string) is not str:
        raise TypeError(
            f"Input must be of type string. Found: type '{type(string)}'")
    if len(string) > 0:
        result_string = string
    else:
        result_string = empty_str
    return result_string

def dict2list(dict_):
    """
    Converts a dict into a list of key-value dictionaries and returns it.

    Parameters
    ----------
    dict_ : dict
        Some dictionary to be converted.

    Returns
    -------
    list_ : list
        Each element is a dict with one key-value pair. These key-value pairs
        are those contained in dict_.
    """
    if type(dict_) != dict:
        raise TypeError(
            f"Input argument must be of type 'dict', found '{type(dict_)}'.")
    list_ = []
    for key, value in dict_.items():
        list_.append({key: value})
    return list_

def list2dict(list_dict):
    """
    Converts a list into a specific dictionary. The list may only contain
    strings or one-element dictionaries. For example [{'a': 'm'}, 'b'] will be
    converted into {'a': 'm', 'b': 'b'}.

    Parameters
    ----------
    list_dict : list or dict
        If it's a list it may only contain strings or one-element dictionaries.

    Returns
    -------
    dict_ : dict
        Strings are mapped to themselves, while one-element dictionaries are
        simply added to this result dictionary.
    """
    # check the given input
    if type(list_dict) not in [list, dict]:
        raise TypeError(
            f"The input argument must be of type 'list' or 'dict'. Found type "
            f"'{type(list_dict)}' however."
        )
    if type(list_dict) is dict:
        # convert the dict to a list, so it can be checked by this function
        list_ = dict2list(copy(list_dict))
    else:
        list_ = copy(list_dict)
    # convert the list to a dictionary
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

def pretty_time_delta(seconds):
    """
    Converts number of seconds into a human friendly time string. Source: https:
    //gist.github.com/thatalextaylor/7408395#file-1-python-pretty-time-delta-py

    Parameters
    ----------
    seconds : float, int
        The given number of seconds to be converted,.

    Returns
    -------
    str
        Human friendly time difference in string format.
    """
    sign_string = "-" if seconds < 0 else ""
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return "%s%dd%dh%dm" % (sign_string, days, hours, minutes)
    elif hours > 0:
        return "%s%dh%dm%ds" % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return "%s%dm%ds" % (sign_string, minutes, seconds)
    else:
        return "%s%ds" % (sign_string, seconds)

def flatten_generator(items):
    """
    Yield items from any nested iterable. This solution is modified from a
    recipe in Beazley, D. and B. Jones. Recipe 4.14, Python Cookbook 3rd Ed.,
    O'Reilly Media Inc. Sebastopol, CA: 2013.

    Parameters
    ----------
    items : Iterable
        A list, tuple, numpy.ndarray, etc. that should be flattened.

    Returns
    -------
    obj[generator]
        Can be translated to a list by applying list(...) on it.
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten_generator(x):
                yield sub_x
        else:
            yield x

def flatten(arg):
    """
    Flattens and returns the given input argument.

    Parameters
    ----------
    arg : list, numpy.ndarray, float, int, None
        The list/array that should be flattened.

    Returns
    -------
    arg_flat : list, None
        The flattened list/numpy.ndarray is the input is not None. Otherwise,
        None is returned.
    """
    arg_type = type(arg)
    if arg is None:
        arg_flat = arg
    elif arg_type in [float, int]:
        arg_flat = [arg]
    else:
        if arg_type not in [list, np.ndarray]:
            raise TypeError(f"The argument must be either None or of type list "
                            f"numpy.ndarray, float or int. Found type "
                            f"{arg_type} however.")
        arg_flat = list(flatten_generator(arg))
    return arg_flat

def process_spatial_coordinates(x=None, y=None, z=None, coords=None,
                                order=('x', 'y', 'z')):
    """
    x : float, int, numpy.ndarray, None, optional
        Positional x-coordinate. When given, the coords-argument must be None.
    y : float, int, numpy.ndarray, None, optional
        Positional y-coordinate. When given, the coords-argument must be None.
    z : float, int, numpy.ndarray, None, optional
        Positional z-coordinate. When given, the coords-argument must be None.
    coords : numpy.ndarray, optional
        Some or all of the coordinates x, y, z concatenated as an array. Each
        row corresponds to one coordinate. For example, row 1 might contain all
        x-coordinates. Which row corresponds to which coordinate is defined via
        the order-argument. When the coords-argument is given, all 3 arguments
        x, y and z must be None.
    order : tuple[str], optional
        Only relevant when coords is given. Defines which row in coords
        corresponds to which coordinate. For example, order=('x', 'y', 'z')
        means that the 1st row are x-coordinates, the 2nd row are y-coords and
        the 3rd row are the z-coordinates.

    Returns
    -------
    coords : numpy.ndarray
        An array with as many columns as coordinates are given, and as many rows
        as points are given. For example if 10 points with x and z coordinates
        are given, then coords would have a shape of (2, 10).
    adjusted_order : list[str]
        Describes which coordinates are described by the rows of the returned
        coords. In the example given above, adjusted_order would be ['x', 'z'].
    """

    # the following check should cover the option that no spatial input is given
    if (x is None) and (y is None) and (z is None) and (coords is None):
        return np.array([]), []

    # convert all single-coordinate inputs to flat numpy arrays
    x = np.array(flatten(x)) if x is not None else None
    y = np.array(flatten(y)) if y is not None else None
    z = np.array(flatten(z)) if z is not None else None

    # derive the number of given coordinate vectors and points
    if coords is not None:
        if not type(coords) is np.ndarray:
            raise TypeError(f"The argument 'coords' must be of type "
                            f"numpy.ndarray. Found {type(coords)} however.")
        else:
            # each row corresponds to one coordinate, so the number of given
            # points is the length of rows
            n_coords, n_points = coords.shape
    else:
        n_points_list = [len(v) for v in [x, y, z] if v is not None]
        n_points_set = set(n_points_list)
        if len(n_points_set) == 1:
            n_coords = len(n_points_list)
            n_points = n_points_list[0]
        else:
            raise RuntimeError(
                f"Found inconsistent lengths in given coordinate "
                f"vectors: {n_points_list}!")

    # derive the coords array and the corresponding order-vector to be returned;
    # note that the repeated if-else clause here should improve readability
    if coords is not None:
        # it is assumed here that the first n_coords elements from the order-
        # vector correspond to the n_coords rows of the given coords-argument
        adjusted_order = list(order[:n_coords])
    else:
        # in this case the order-vector might have to be trimmed; for example if
        # x and z are given, the 'y' from the order vector has to be removed
        coords = np.zeros((n_coords, n_points))
        adjusted_order = []
        row_idx = 0
        for v in order:
            if eval(v) is not None:
                adjusted_order.append(v)
                coords[row_idx, :] = eval(v)
                row_idx += 1

    return coords, adjusted_order

def translate_prms_def(prms_def_given):
    """
    Translates the prms_def argument which is used by several sub-modules (e.g.
    ForwardModelBase, NoiseModelBase, PriorBase) into a default format. The
    prms_def argument specifies the local/global names of the parameters used by
    a sub-module.

    Parameters
    ----------
    prms_def_given : {str, list, dict}
        Either a single string, a dictionary with global names as keys and local
        names as values, or a list, the elements of which are either strings or
        1-element dictionaries, where the latter would again contain one global
        name as key and one local name as value. Valid examples are: 'sigma',
        ['sigma'], ['sigma', 'beta'], ['sigma', {'beta': 'b'}],
        {'sigma': 'sigma', 'beta': 'b'}.

    Returns
    -------
    prms_def : dict
        Contains global names as keys and local names as values.
    prms_dim : int
        The number of items in prms_def.
    """
    prms_def_copy = copy(prms_def_given)
    if type(prms_def_copy) is dict:
        prms_def = list2dict(prms_def_copy)
    else:
        prms_def = list2dict(make_list(prms_def_copy))
    prms_dim = len(prms_def)
    return prms_def, prms_dim
