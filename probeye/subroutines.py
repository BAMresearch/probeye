# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# standard library imports
from copy import copy
from typing import Iterable
import os
import sys

# third party imports
import numpy as np
from loguru import logger
from functools import partial

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
    int
        The length of the given list/tuple etc. or 1, if obj has no __len__
        attribute; the latter case is mostly intended for scalar numbers

    """
    if hasattr(obj, '__len__'):
        # the following check is necessary, since the len-function applied to a
        # numpy array of format numpy.array(1) results in a TypeError
        if type(obj) is np.ndarray:
            if not obj.shape:
                return 1
            else:
                return len(obj)
        else:
            return len(obj)
    else:
        return 1

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

def underlined_string(string, symbol="=", n_empty_start=1, n_empty_end=1):
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

def process_spatiotemporal_coordinates(x=None, y=None, z=None, t=None,
                                       coords=None, order=('x', 'y', 'z', 't')):
    """
    x : float, int, numpy.ndarray, None, optional
        Positional x-coordinate. When given, the coords-argument must be None.
    y : float, int, numpy.ndarray, None, optional
        Positional y-coordinate. When given, the coords-argument must be None.
    z : float, int, numpy.ndarray, None, optional
        Positional z-coordinate. When given, the coords-argument must be None.
    t : float, int, numpy.ndarray, None, optional
            Points in time. When given, coords must be None.
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
    if (x is None) and (y is None) and (z is None) and \
            (t is None) and (coords is None):
        return np.array([]), []

    # convert all single-coordinate inputs to flat numpy arrays
    x = np.array(flatten(x)) if x is not None else None
    y = np.array(flatten(y)) if y is not None else None
    z = np.array(flatten(z)) if z is not None else None
    t = np.array(flatten(t)) if t is not None else None

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
        n_points_list = [len(v) for v in [x, y, z, t] if v is not None]
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

def print_probeye_header(width=100, header_file="probeye.txt", version='1.0.11',
                         margin=5, h_symbol="=", v_symbol="#", use_logger=True):
    """
    Prints the probeye header which is printed, when an inference problem is
    set up. Mostly just nice to have. The only useful information it contains
    is the version number of the package.

    Parameters
    ----------
    width : int, optional
        The width (i.e., number of characters) the header should have.
    header_file : str, optional
        Relative path (with respect to this file) to the txt-file that contains
        the probeye letters.
    version : str, optional
        States the probeye version; this should be identical to the version
        stated in setup.cfg; however, the version cannot be read dynamically,
        since the setup.cfg is not available after installing the package.
    margin : int, optional
        Minimum number of blank spaces at the header margins.
    h_symbol : str, optional
        The symbol used to 'draw' the horizontal frame line.
    v_symbol : str, optional
        The symbol used to 'draw' the vertical frame line.
    use_logger : bool, optional
        When True, the header will be logged, otherwise just printed.
    """

    # define the full paths of the given files
    dir_path = os.path.dirname(__file__)
    header_file = os.path.join(dir_path, header_file)

    # read in the big probeye letters
    with open(header_file, 'r') as f:
        content = f.readlines()
    # this is the width of the read in 'probeye' in terms of number of chars;
    # note that all lines (should) have the same length
    width_probeye = len(content[0]) - 1

    # this string should coincide with the one given in setup.cfg; however, it
    # cannot be read dynamically, since the setup.cfg is not available after
    # installing the package
    description = 'A general framework for setting up parameter '\
                  'estimation problems.'

    subtitle = f"Version {version} - {description}"
    width_subtitle = len(subtitle)

    # choose a width so that the margin on one side is at least 'margin'
    width_used = max((width, width_probeye + 2 * (margin + 1),
                      width_subtitle + 2 * margin + 1))

    # assemble the header
    outer_frame_line = f"{v_symbol} {h_symbol * (width_used - 4)} {v_symbol}"
    inner_frame_line = f"{v_symbol}{' ' * (width_used - 2)}{v_symbol}"
    lines = [outer_frame_line, inner_frame_line]
    for line in content:
        clean_line = line.replace('\n', '')
        lines.append(f"{v_symbol}{clean_line:^{width_used - 2}s}{v_symbol}")
    lines.append(inner_frame_line)
    lines.append(outer_frame_line)
    lines.append(inner_frame_line)
    lines.append(f"{v_symbol}{subtitle:^{width_used - 2}s}{v_symbol}")
    lines.append(inner_frame_line)
    lines.append(outer_frame_line)

    # log or print the header
    if use_logger:
        print("")
        for line in lines:
            logger.info(line)
    else:
        print('\n' + '\n'.join(lines))

def logging_setup(log_level_stdout='INFO', log_level_file='DEBUG',
                  log_format=None, log_file=None, overwrite_log_file=True,
                  **kwargs):
    """
    Sets up the loguru logger for listening to the inference problem.

    Parameters
    ----------
    log_level_stdout : {'DEBUG', 'INFO', 'WARNING', 'ERROR'}, optional
        Defines the level of the logging output to stdout.
    log_level_file : {'DEBUG', 'INFO', 'WARNING', 'ERROR'}, optional
        Defines the level of the logging output to a log file.
    log_format : None, str, optional
        A format string defining the logging output. If this argument is
        set to None, a default format will be used.
    log_file : None, str, optional
        Path to the log-file, if the logging should be printed to file. If
        None is given, no logging-file will be created.
    overwrite_log_file : bool, optional
        When True, a specified log-file will be overwritten. Otherwise,
        the generated logging will appended to a given log-file.
    kwargs : dict
        Additional keyword arguments passed to logger.add (for file and stdout).
    """
    if not log_format:
        log_format =\
            ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
             '<level>{level: <8}</level> | '
             '<level>{message:100s}</level> | '
             '<cyan>{name}</cyan>:'
             '<cyan>{function}</cyan>:'
             '<cyan>{line}</cyan>')
    logger.remove()  # just in case there still exists another logger
    logger.add(sys.stdout, format=log_format, level=log_level_stdout, **kwargs)
    if log_file:
        if os.path.isfile(log_file) and overwrite_log_file:
            os.remove(log_file)
        logger.add(
            log_file, format=log_format, level=log_level_file, **kwargs)

def stream_to_logger(log_level):
    """
    Returns a stream-object that can be used to redirect a function's print
    output to the logger. Taken from the section 'Capturing standard stdout ...'
    of https://loguru.readthedocs.io/en/stable/resources/recipes.html.

    Parameters
    ----------
    log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR'}, optional
        Defines the log level the streamed output will be associated with.

    Returns
    -------
    obj[StreamToLogger]
        This object should be used as follows:
        import contextlib
        with contextlib.redirect_stdout(stream_to_logger('INFO')):
            <function that prints something>
    """

    class StreamToLogger:
        def __init__(self, level):
            self._level = level

        def write(self, buffer):
            for line in buffer.rstrip().splitlines():
                logger.opt(depth=1).log(self._level, line.rstrip())

    return StreamToLogger(log_level)

def print_dict_in_rows(d, printer=print, sep="=", val_fmt=None):
    """
    Prints a dictionary with key-value pairs in rows.

    Parameters
    ----------
    d : dict
        The dictionary to print.
    printer : callable, optional
        Function used for printing. For example 'print' or 'logger.info'.
    sep : str, optional
        The character printed between key and value.
    val_fmt : str, optional
        A format string used for printing the dictionary's values.
    """
    n = max([len(key) for key in d.keys()])
    for key, val in d.items():
        if val_fmt:
            printer(f"{key:{n + 1}s} {sep} {val:{val_fmt}}")
        else:
            printer(f"{key:{n + 1}s} {sep} {val}")

def add_index_to_tex_prm_name(tex, index):
    """
    Adds a lower index to a parameter's tex-name. This function is intended for
    vector-valued parameters. For example: ('$a$', 1) -> '$a_1$'.

    Parameters
    ----------
    tex : str
        The tex-string to be modified.
    index : int
        The index to be added as a lower index to tex.

    Returns
    -------
    tex_mod : str
        The tex-string with included index.
    """

    # the math-model '$' should appear twice in the string
    check_1 = tex.count("$") == 2
    # the index is only added in tex-fashion
    # if no indexes are present already
    check_2 = not ("_" in tex)
    check_3 = not ("^" in tex)

    if check_1 and check_2 and check_3:
        tex_list = tex.split("$")
        # since it was checked that there are exactly 2 '$'-signs in tex, the
        # tex_list has 3 elements, with the middle one being the string enclosed
        # by the two '$'-signs
        tex_list[1] = tex_list[1] + f'_{index}'
        tex_mod = '$'.join(tex_list)
    else:
        # if not all checks are passed, the index is added in a way, that does
        # not expect anything from the given tex-string
        tex_mod = tex + f" ({index})"

    return tex_mod

def check_for_uninformative_priors(problem):
    """
    Checks if all priors defined within a given InferenceProblem are not
    uninformative.

    Parameters
    ----------
    problem : InferenceProblem
        The given problem to check.
    """
    for prior_name, prior_template in problem.priors.items():
        if prior_template.prior_type == 'uninformative':
            raise RuntimeError(f"The prior '{prior_name}' is uninformative,"
                               f" which cannot be used by the requested "
                               f"solver. You could change it to a "
                               f"uniform-prior on a specified interval to "
                               f"solver this problem.")

def get_dictionary_depth(d):
    """
    Computes the depth of a nested dictionary recursively. Modified from
    https://www.geeksforgeeks.org/python-find-depth-of-a-dictionary/.

    Parameters
    ----------
    d : dict
        The dictionary, the depth should be computed of.

    Returns
    -------
    int
        The depth of the given dictionary d.
    """
    if isinstance(d, dict):
        return 1 + (max(map(get_dictionary_depth, d.values())) if d else 0)
    return 0


def compute_reduction_array(array):
    n = array.shape[0]
    a = np.eye(n)
    rows_to_remove = []
    for i, reference_row in enumerate(array):
        rows_to_add = []
        for j, row_to_check in enumerate(array[i+1:], start=i+1):
            if np.allclose(row_to_check, reference_row):
                rows_to_add.append(j)
        for row_idx in rows_to_add:
            a[i] += a[row_idx]
        a[i] /= 1 + len(rows_to_add)
        rows_to_remove += rows_to_add
    a = np.delete(a, rows_to_remove, axis=0)
    return a, rows_to_remove
