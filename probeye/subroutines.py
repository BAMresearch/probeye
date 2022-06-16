# standard library imports
import types
from copy import copy, deepcopy
from typing import Union, List, Tuple, Any, Optional, Generator, Callable
from typing import TYPE_CHECKING
import os
import sys

# third party imports
import numpy as np
from tripy.utils import correlation_function
from tripy.utils import correlation_matrix
from loguru import logger

# local imports
from probeye import __version__

# local imports for type checking
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


def len_or_one(obj: Any) -> int:
    """
    Returns the length of an object or 1 if no length is defined.

    Parameters
    ----------
    obj
        Most of the time this will be a list/tuple or a single scalar number. But
        generally, it can be anything.

    Returns
    -------
        The length of the given list/tuple etc. or 1, if obj has no __len__-attribute;
        the latter case is mostly intended for scalar numbers.

    """
    if hasattr(obj, "__len__"):
        # the following check is necessary, since the len-function applied to a numpy
        # array of format numpy.array(1) results in a TypeError
        if type(obj) is np.ndarray:
            if not obj.shape:
                return 1
            else:
                return len(obj)
        else:
            return len(obj)
    else:
        return 1


def make_list(arg: Any) -> list:
    """
    Converts a given argument into a list, if it is not a list or tuple. The typical use
    case for this method is to convert a single string into a list with this string as
    its only element, e.g. make_list('sigma') = ['sigma'].

    Parameters
    ----------
    arg
        Essentially anything. Most of the time this might be a string or a list of
        strings or a tuple of strings.

    Returns
    -------
    new_arg
        Either arg if it is of type list or tuple, or a list with arg as its only
        element if arg is not of type list or tuple.
    """
    if type(arg) in [list, tuple]:
        new_arg = arg
    else:
        new_arg = [copy(arg)]
    return new_arg


def underlined_string(
    string: str, symbol: str = "=", n_empty_start: int = 1, n_empty_end: int = 1
) -> str:
    """
    Adds a line made of 'symbol'-characters under a given string and returns it.

    Parameters
    ----------
    string
        The string that should be underlined.
    symbol
        A single character the line should be 'made' of.
    n_empty_start
        Number of empty lines added before the underlined string.
    n_empty_end
        Number of empty lines added after the underlined string.

    Returns
    -------
    result_string
        The generated string representing an underlined string, possibly with empty
        lines added before/after.
    """
    n_chars = len(string)
    underline_string = n_chars * symbol
    empty_lines_start = n_empty_start * "\n"
    empty_lines_end = n_empty_end * "\n"
    result_string = string + "\n" + underline_string
    result_string = empty_lines_start + result_string + empty_lines_end
    return result_string


def titled_table(
    title_str: str,
    table_str: str,
    symbol: str = "-",
    n_empty_start: int = 1,
    n_empty_end: int = 0,
) -> str:
    """
    Adds an underlined title string to a given table string. The line, that underlines
    the title will be as long as the longest line of the table.

    Parameters
    ----------
    title_str
        The title to be put on top of the table.
    table_str
        The string representing the table. For example generated using tabulate.
    symbol
        A single character the line should be 'made' of.
    n_empty_start
        Number of empty lines added before the title.
    n_empty_end
        Number of empty lines added after the table string.

    Returns
    -------
    result_string
        An underlined title, followed by a table.
    """
    # get the number of characters in the given table's longest line
    max_line_length = max([len(line) for line in table_str.split("\n")])
    # now, simply concatenate the different lines
    result_string = (
        n_empty_start * "\n"
        + title_str
        + "\n"
        + max_line_length * symbol
        + "\n"
        + table_str
        + "\n"
        + n_empty_end * "\n"
    )
    return result_string


def replace_string_chars(
    string: str, replace: Optional[dict] = None, remove: Optional[list] = None
) -> str:
    """
    Removes and replaces characters from a given string according to the input.

    Parameters
    ----------
    string
        The string to be modified.
    replace
        The keys are the characters to be replaced, the values are stating their
        replacements.
    remove
        A list of characters to be removed from the given string.

    Returns
    -------
     string
        The modified string with removed/replaced characters.
    """
    # first, do the replacements
    if replace is not None:
        for char, replacement in replace.items():
            string = string.replace(char, replacement)
    # finally, remove characters as requested
    if remove is not None:
        for char in remove:
            string = string.replace(char, "")
    return string


def simplified_list_string(list_: list) -> str:
    """
    Given a list, it returns the string returned by its __str__ method, where some
    characters are removed for a slightly easier to read printout. Example: the list
    ['a', 1.2] is usually printed as '['a', 1.2]'. Here, it will be converted into the
    string 'a, 1.2' (no brackets, and no quotes).

    Parameters
    ----------
    list_
        Some list to be printed.

    Returns
    -------
    simplified_list_str
        The list_'s __str__ method's return string without brackets and quotes.
    """
    simplified_list_str = replace_string_chars(str(list_), remove=["[", "]", "'"])
    return simplified_list_str


def simplified_dict_string(dict_: dict) -> str:
    """
    Given a dictionary, it returns the string returned by its __str__ method, where some
    characters are removed for a slightly easier to read printout. For example: the dict
    {'a': 1.2} is usually printed as '{'a': 1.2, 'b': 2.1}'. Here, it will be converted
    into the string 'a=1.2, b=2.1'.

    Parameters
    ----------
    dict_
        Some dictionary to be printed.

    Returns
    -------
    simplified_dict_str
        Modified version of dict_'s __str__ method's return string (no quotes, no braces
        and the colon will be replaced with an equal sign.
    """
    simplified_dict_str = replace_string_chars(
        str(dict_), remove=["{", "}", "'"], replace={": ": "="}
    )
    return simplified_dict_str


def sub_when_empty(string: str, empty_str: str = "-") -> str:
    """
    Just returns a given string if it is not empty. If it is empty though, a default
    string is returned instead.

    Parameters
    ----------
    string
        The string to check if it is empty or not
    empty_str
        The string to be returned if the given string is empty

    Returns
    -------
    result_string
        Either the given string (when 'string' is not empty) or the empty_str (when
        'string' is empty)
    """
    if type(string) is not str:
        raise TypeError(f"Input must be of type string. Found: type '{type(string)}'")
    if len(string) > 0:
        result_string = string
    else:
        result_string = empty_str
    return result_string


def dict2list(dict_: dict) -> list:
    """
    Converts a dict into a list of key-value dictionaries and returns it.

    Parameters
    ----------
    dict_
        Some dictionary to be converted.

    Returns
    -------
    list_
        Each element is a dict with one key-value pair. These key-value pairs
        are those contained in dict_.
    """
    if type(dict_) != dict:
        raise TypeError(
            f"Input argument must be of type 'dict', found '{type(dict_)}'."
        )
    list_ = []
    for key, value in dict_.items():
        list_.append({key: value})
    return list_


def list2dict(list_dict: Union[list, dict]) -> dict:
    """
    Converts a list into a specific dictionary. The list may only contain strings or
    one-element dictionaries. For example [{'a': 'm'}, 'b'] will be converted into
    {'a': 'm', 'b': 'b'}.

    Parameters
    ----------
    list_dict
        If it's a list it may only contain strings or one-element dictionaries.

    Returns
    -------
    dict_
        Strings are mapped to themselves, while one-element dictionaries are simply
        added to this result dictionary.
    """
    # check the given input
    if type(list_dict) not in [list, dict]:
        raise TypeError(
            f"The input argument must be of type 'list' or 'dict'. Found type "
            f"'{type(list_dict)}' however."
        )
    if type(list_dict) is dict:
        # convert the dict to a list, so it can be checked by this function
        list_ = dict2list(copy(list_dict))  # type: Union[list, dict]
    else:
        list_ = copy(list_dict)
    # convert the list to a dictionary
    dict_ = {}  # type: dict
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


def pretty_time_delta(seconds: Union[float, int]) -> str:
    """
    Converts number of seconds into a human friendly time string. Source: https:
    //gist.github.com/thatalextaylor/7408395#file-1-python-pretty-time-delta-py

    Parameters
    ----------
    seconds
        The given number of seconds to be converted,.

    Returns
    -------
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


def flatten_generator(items: Union[list, np.ndarray]) -> Generator:
    """
    Yield items from any nested iterable. This solution is modified from a recipe in
    Beazley, D. and B. Jones. Recipe 4.14, Python Cookbook 3rd Ed., O'Reilly Media Inc.
    Sebastopol, CA: 2013.

    Parameters
    ----------
    items
        A list, tuple, numpy.ndarray, etc. that should be flattened.

    Returns
    -------
    obj[generator]
        Can be translated to a list by applying list(...) on it.
    """
    for x in items:
        if type(x) in [list, tuple, np.ndarray] and not isinstance(x, (str, bytes)):
            for sub_x in flatten_generator(x):
                yield sub_x
        else:
            yield x


def flatten(arg: Union[list, np.ndarray, float, int, None]) -> Union[list, None]:
    """
    Flattens and returns the given input argument.

    Parameters
    ----------
    arg
        The list/array that should be flattened.

    Returns
    -------
    arg_flat
        The flattened list/numpy.ndarray is the input is not None. Otherwise, None is
        returned.
    """
    arg_type = type(arg)
    if arg is None:
        arg_flat = arg
    elif arg_type in [float, int]:
        arg_flat = [arg]
    elif arg_type in [list, np.ndarray]:
        arg_flat = list(flatten_generator(arg))  # type: ignore
    else:
        raise TypeError(
            f"The argument must be either None or of type list numpy.ndarray, "
            f"float or int. Found type '{arg_type}' however."
        )
    return arg_flat


def process_spatiotemporal_coordinates(
    x: Union[float, int, np.ndarray, list, None] = None,
    y: Union[float, int, np.ndarray, list, None] = None,
    z: Union[float, int, np.ndarray, list, None] = None,
    t: Union[float, int, np.ndarray, list, None] = None,
    coords: Optional[np.ndarray] = None,
    order: tuple = ("x", "y", "z", "t"),
) -> Tuple[np.ndarray, List[str]]:
    """
    Converts given spatial data from a flexible format into a standardized format.

    Parameters
    ----------
    x
        Positional x-coordinate. When given, the coords-argument must be None.
    y
        Positional y-coordinate. When given, the coords-argument must be None.
    z
        Positional z-coordinate. When given, the coords-argument must be None.
    t
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
    coords
        An array with as many columns as coordinates are given, and as many rows as
        points are given. For example if 10 points with x and z coordinates are given,
        then coords would have a shape of (2, 10).
    adjusted_order
        Describes which coordinates are described by the rows of the returned coords.
        In the example given above, adjusted_order would be ['x', 'z'].
    """

    # the following check should cover the option that no spatial input is given
    if (x is None) and (y is None) and (z is None) and (t is None) and (coords is None):
        return np.array([]), []

    # convert all single-coordinate inputs to flat numpy arrays
    x = np.array(flatten(x)) if x is not None else None
    y = np.array(flatten(y)) if y is not None else None
    z = np.array(flatten(z)) if z is not None else None
    t = np.array(flatten(t)) if t is not None else None

    # derive the number of given coordinate vectors and points
    if coords is not None:
        if not type(coords) is np.ndarray:
            raise TypeError(
                f"The argument 'coords' must be of type numpy.ndarray. Found "
                f"{type(coords)} however."
            )
        else:
            # each row corresponds to one coordinate, so the number of given points is
            # the length of rows
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
                f"vectors: {n_points_list}!"
            )

    # derive the coords array and the corresponding order-vector to be returned; note
    # that the repeated if-else clause here should improve readability
    if coords is not None:
        # it is assumed here that the first n_coords elements from the order-vector
        # correspond to the n_coords rows of the given coords-argument
        adjusted_order = list(order[:n_coords])
    else:
        # in this case the order-vector might have to be trimmed; for example if x and
        # z are given, the 'y' from the order vector has to be removed
        coords = np.zeros((n_coords, n_points))
        adjusted_order = []
        row_idx = 0
        for v in order:
            if eval(v) is not None:
                adjusted_order.append(v)
                coords[row_idx, :] = eval(v)
                row_idx += 1

    return coords, adjusted_order


def translate_prms_def(prms_def_given: Union[str, list, dict]) -> Tuple[dict, int]:
    """
    Translates the prms_def argument which is used by several sub-modules (e.g.
    ForwardModelBase, NoiseModelBase, PriorBase) into a default format. The prms_def-
    argument specifies the local/global names of the parameters used by a sub-module.

    Parameters
    ----------
    prms_def_given
        Either a single string, a dictionary with global names as keys and local names
        as values, or a list, the elements of which are either strings or 1-element
        dictionaries, where the latter would again contain one global name as key and
        one local name as value. Valid examples are: 'sigma', ['sigma'], ['sigma',
        'beta'], ['sigma', {'beta': 'b'}], {'sigma': 'sigma', 'beta': 'b'}.

    Returns
    -------
    prms_def
        Contains global names as keys and local names as values.
    prms_dim
        The number of items in prms_def.
    """
    prms_def_copy = copy(prms_def_given)
    if type(prms_def_copy) is dict:
        prms_def = list2dict(prms_def_copy)
    else:
        prms_def = list2dict(make_list(prms_def_copy))
    prms_dim = len(prms_def)
    return prms_def, prms_dim


def print_probeye_header(
    width: int = 100,
    header_file: str = "probeye.txt",
    version: str = __version__,
    margin: int = 5,
    h_symbol: str = "=",
    v_symbol: str = "#",
    use_logger: bool = True,
):
    """
    Prints the probeye header which is printed, when an inverse problem is set up.
    Mostly just nice to have. The only useful information it contains is the version
    number of the package.

    Parameters
    ----------
    width
        The width (i.e., number of characters) the header should have.
    header_file
        Relative path (with respect to this file) to the txt-file that contains the
        probeye letters.
    version
        States the probeye version; this should be identical to the version stated in
        setup.cfg; however, the version cannot be read dynamically, since the setup.cfg
        is not available after installing the package.
    margin
        Minimum number of blank spaces at the header margins.
    h_symbol
        The symbol used to 'draw' the horizontal frame line.
    v_symbol
        The symbol used to 'draw' the vertical frame line.
    use_logger
        When True, the header will be logged, otherwise just printed.
    """

    # define the full paths of the given files
    dir_path = os.path.dirname(__file__)
    header_file = os.path.join(dir_path, header_file)

    # read in the big probeye letters
    with open(header_file, "r") as f:
        content = f.readlines()
    # this is the width of the read in 'probeye' in terms of number of chars; note that
    # all lines (should) have the same length
    width_probeye = len(content[0]) - 1

    # this string should coincide with the one given in setup.cfg; however, it cannot be
    # read dynamically since the setup.cfg is not available after installing the package
    description = "A general framework for setting up parameter " "estimation problems."
    subtitle = f"Version {version} - {description}"
    width_subtitle = len(subtitle)

    # choose a width so that the margin on one side is at least 'margin'
    width_used = max(
        (width, width_probeye + 2 * (margin + 1), width_subtitle + 2 * margin + 1)
    )

    # assemble the header
    outer_frame_line = f"{v_symbol} {h_symbol * (width_used - 4)} {v_symbol}"
    inner_frame_line = f"{v_symbol}{' ' * (width_used - 2)}{v_symbol}"
    lines = [outer_frame_line, inner_frame_line]
    for line in content:
        clean_line = line.replace("\n", "")
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
        print("\n" + "\n".join(lines))


def logging_setup(
    log_level_stdout: str = "INFO",
    log_level_file: str = "DEBUG",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    overwrite_log_file: bool = True,
    **kwargs,
):
    """
    Sets up the loguru logger for listening to the inference problem.

    Parameters
    ----------
    log_level_stdout
        Defines the level of the logging output to stdout. Common choices are 'DEBUG',
        'INFO', 'WARNING', and 'ERROR'.
    log_level_file
        Defines the level of the logging output to a log file. Common choices are again
        'DEBUG', 'INFO', 'WARNING', and 'ERROR'.
    log_format
        A format string defining the logging output. If this argument is
        set to None, a default format will be used.
    log_file
        Path to the log-file, if the logging should be printed to file. If
        None is given, no logging-file will be created.
    overwrite_log_file
        When True, a specified log-file will be overwritten. Otherwise,
        the generated logging will appended to a given log-file.
    kwargs
        Additional keyword arguments passed to logger.add (for file and stdout).
    """
    if not log_format:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message:100s}</level> | "
            "<cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan>"
        )
    logger.remove()  # just in case there still exists another logger
    logger.add(sys.stdout, format=log_format, level=log_level_stdout, **kwargs)
    if log_file:
        if os.path.isfile(log_file) and overwrite_log_file:
            os.remove(log_file)
        logger.add(log_file, format=log_format, level=log_level_file, **kwargs)


class StreamToLogger:
    """This class is required by stream_to_logger defined right below."""

    def __init__(self, level):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())


def stream_to_logger(log_level: str) -> StreamToLogger:
    """
    Returns a stream-object that can be used to redirect a function's print output to
    the logger. Taken from the section 'Capturing standard stdout ...' of
    https://loguru.readthedocs.io/en/stable/resources/recipes.html.

    Parameters
    ----------
    log_level
        Defines the log level the streamed output will be associated with. Common
        choices are 'DEBUG', 'INFO', 'WARNING', and 'ERROR'.

    Returns
    -------
        This object should be used as follows:
        import contextlib
        with contextlib.redirect_stdout(stream_to_logger('INFO')):
            <function that prints something>
    """
    return StreamToLogger(log_level)


def print_dict_in_rows(
    d: dict, printer: Callable = print, sep: str = "=", val_fmt: Optional[str] = None
):
    """
    Prints a dictionary with key-value pairs in rows.

    Parameters
    ----------
    d
        The dictionary to print.
    printer
        Function used for printing. For example 'print' or 'logger.info'.
    sep
        The character printed between key and value.
    val_fmt
        A format string used for printing the dictionary's values.
    """
    n = max([len(key) for key in d.keys()])
    for key, val in d.items():
        if val_fmt:
            printer(f"{key:{n + 1}s} {sep} {val:{val_fmt}}")
        else:
            printer(f"{key:{n + 1}s} {sep} {val}")


def add_index_to_tex_prm_name(tex: str, index: int) -> str:
    """
    Adds a lower index to a parameter's tex-name. This function is intended for vector-
    valued parameters. For example: ('$a$', 1) -> '$a_1$'.

    Parameters
    ----------
    tex
        The tex-string to be modified.
    index
        The index to be added as a lower index to tex.

    Returns
    -------
    tex_mod
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
        # since it was checked that there are exactly 2 '$'-signs in tex, the tex_list
        # has 3 elements, with the middle one being the string enclosed by the two
        # '$'-signs
        tex_list[1] = tex_list[1] + f"_{index}"
        tex_mod = "$".join(tex_list)
    else:
        # if not all checks are passed, the index is added in a way, that does not
        # expect anything from the given tex-string
        tex_mod = tex + f" ({index})"

    return tex_mod


def check_for_uninformative_priors(problem: "InverseProblem"):
    """
    Checks if all priors defined within a given InverseProblem are not uninformative.

    Parameters
    ----------
    problem
        The given problem to check.
    """
    for prior_name, prior_template in problem.priors.items():
        if prior_template.prior_type == "uninformative":
            raise RuntimeError(
                f"The prior '{prior_name}' is uninformative,"
                f" which cannot be used by the requested "
                f"solver. You could change it to a "
                f"uniform-prior on a specified interval to "
                f"solver this problem."
            )


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


def compute_reduction_array(array: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Given a square array with potentially duplicate rows, this method computes an
    array, that, if matrix-multiplied with the given array, leaves all unique rows
    un-modified, but averages all duplicate rows, writes the result in the first
    row of their appearance, and removes all other rows of the duplicate ones. For
    example:

    [[1, 0, 0]            [[0.5, 0.5, 0]
     [1, 0, 0]     -->     [ 0 ,  0 , 1]]
     [0, 0, 1]]

    Returns
    -------
    reduction_array
        The derived reduction array for the given array 'array'.
    rows_to_remove
        A list that contains the row indices of duplicate rows.
    """
    n = array.shape[0]
    reduction_array = np.eye(n)
    rows_to_remove = []
    for i, reference_row in enumerate(array):
        rows_to_add = []
        for j, row_to_check in enumerate(array[i + 1 :], start=i + 1):
            if np.allclose(row_to_check, reference_row):
                rows_to_add.append(j)
        for row_idx in rows_to_add:
            reduction_array[i] += reduction_array[row_idx]
        reduction_array[i] /= 1 + len(rows_to_add)
        rows_to_remove += rows_to_add
    reduction_array = np.delete(reduction_array, rows_to_remove, axis=0)
    return reduction_array, rows_to_remove


def incrementalize(
    v_in: Union[np.ndarray, list], eps: float = 1e-12
) -> Tuple[Union[np.ndarray, list], Callable, bool]:
    """
    Given some vector, this function rearranges the elements so that they are ascending,
    and removes duplicate elements.

    Parameters
    ----------
    v_in
        The vector to be processed.
    eps
        A threshold value, that defines when two values are considered identical.

    Returns
    -------
    v_out
        The processed input vector.
    fun
        See the explanations in the function docstring below.
    is_incremental
        If v_in has no duplicate elements and is already in ascending order, this flag
        is True. Otherwise, it is False.
    """

    # check if the given vector is already sorted
    n = len(v_in)
    is_incremental = all(v_in[i] < v_in[i + 1] for i in range(n - 1))

    # sort the vector if necessary
    if is_incremental:
        v_out = v_in
        fun = lambda x: x  # identity function
    else:

        # sort v_in and remove duplicate elements; the result is v_out
        idx_sorted = np.argsort(v_in)
        v_sorted = v_in[idx_sorted]
        v_diff = np.diff(v_sorted)
        indices_to_remove = [i for i, vi in enumerate(v_diff) if np.abs(vi) < eps]
        m = len(indices_to_remove)
        v_out = np.delete(v_sorted, indices_to_remove)

        # this list will contain triples; the first two entries of a triple (lets call
        # them t1 and t2) refer to the indices of v_out in that sense, that v_out[t1:t2]
        # have identical values; the last entry of a triple is just t2-t3 and is needed
        # for the averaging operation performed by the function f
        triples = []

        # derive the triples
        i = 0
        while i < m:
            j = i
            while j + 1 < m and (indices_to_remove[j + 1] - indices_to_remove[j] == 1):
                j += 1
            idx_1 = indices_to_remove[i]
            idx_2 = indices_to_remove[j] + 1 + 1
            triples.append((idx_1, idx_2, idx_2 - idx_1))
            i = j + 1

        def fun(w: np.ndarray) -> np.ndarray:
            """
            Performs an averaging operations over slices of w that correspond to
            duplicate elements in the sorted version of v_in. For example, in the case
            of v_sorted = [1, 2, 2, 3] and w = [8, 3, 6, 0], this function would return
            ws = [8, (3+6)/2, 3] = [8, 4.5, 3].

            Parameters
            ----------
            w
                The input vector which is understood as the picture of v_sorted.

            Returns
            -------
            ws
                The processed input vector.
            """
            ws = w[idx_sorted]
            nr = 0  # number of already removed elements
            for t in triples:
                ws[t[0] - nr] = np.sum(ws[t[0] - nr : t[1] - nr]) / t[2]
                ws = np.delete(ws, np.s_[t[0] - nr + 1 : t[1] - nr])
                nr += t[2] - 1
            return ws

    return v_out, fun, is_incremental


def extract_true_values(true_values: dict, var_names: List[str]) -> np.ndarray:
    """
    Returns the values from a true_values dictionary in the order given by var_names.

    Parameters
    ----------
    true_values
        Keys are parameter names and values are either floats, integers or 1D-arrays.
    var_names
        List stating the order in which the true values should be returned.
    """
    true_values_vector = []
    for var_name in var_names:
        if var_name in true_values:
            true_values_vector.append(float(true_values[var_name]))
        else:
            # this case is the reason why this function was written; it arises when
            # vector-valued parameters are used; if for example 'mb' is a 2-element
            # parameter, then var_names will contain 'mb_1' (not starting at 0) and
            # 'mb_2' but true_values will just contain 'mb'
            var_name_no_index = var_name[::-1].split("_", 1)[1][::-1]
            idx = int(var_name[::-1].split("_", 1)[0]) - 1
            true_values_vector.append(float(true_values[var_name_no_index][idx]))
    return np.array(true_values_vector)


def translate_simple_correlation(corr_string: str) -> dict:
    """
    Translates a string describing a correlation definition with standard correlation
    variables (x, y, z, t) in a dictionary that can be used for correlation_info.

    Parameters
    ----------
    corr_string
        A string like 'T1:xy'. This string contains 3 elements. 'T1' can be any other
        string without a colon ':'. Afterwards, there must follow a colon ':'. Then, the
        last element is a number of non-repeating characters from ('x', 'y', 'z', 't').

    Returns
    -------
    corr_dict
        In the example, 'T1:xy' would be translated to {'T1': {'x': 'x', 'y': 'y'}}.
        This dictionary can be used for the 'correlation_info' argument when adding new
        experiments to an InverseProblem.
    """
    if not (":" in corr_string):
        raise ValueError(f"The given 'corr_string' ('{corr_string}') contains no ':'!")
    sensors = corr_string.split(":")
    if len(sensors) > 2:
        raise ValueError(
            f"The given 'corr_string' ('{corr_string}') contains more than one ':'!"
        )
    corr_dict = {sensors[0]: {}}  # type: dict
    for character in sensors[1]:
        if character not in ["x", "y", "z", "t"]:
            raise ValueError(
                f"The given 'corr_string' ('{corr_string}') does contain a non-standard"
                f" variable ('{character}' instead of 'x', 'y', 'z', 't') after the "
                f"colon ':'!"
            )
        corr_dict[sensors[0]][character] = character
    return corr_dict


def get_global_name(local_name_given: str, prms_def: dict) -> str:
    """
    Gets the global name of a parameter based on a given local name from a 'prms_def'
    dictionary, which holds <global_name>: <local name> items.

    Parameters
    ----------
    local_name_given
        The given local name of some parameter.
    prms_def
        A dict holding global names as keys and local names (all unique) as values.

    Returns
    -------
    global_name
        The global name to the given local name taken from prms_def.
    """
    for global_name, local_name in prms_def.items():
        if local_name == local_name_given:
            return global_name
    raise RuntimeError(f"Given local name '{local_name_given}' not found!")


def translate_number_string(s: str) -> float:
    """
    Translates a given string that describes a number or infinity into a float.

    Parameters
    ----------
    s
        A string that describes a number or +/- infinity.

    Returns
    -------
        Either the number described by the string, or +/- np.infty in the case of an
        infinity value.
    """
    if s in ["oo", "+oo"]:
        return np.infty
    elif s == "-oo":
        return -np.infty
    else:
        return float(s)


def count_intervals(domain_string: str) -> int:
    """
    Counts the number of 1D intervals given by a domain string, i.e., something like
    '(0, 1) [0, 1] (0, 1] (0, 1)'.

    Parameters
    ----------
    domain_string
        A string describing one or more intervals. Valid values are for example '[0, 1]'
        or '(-1,1)(-1,5]'.

    Returns
    -------
        The number of intervals given via the domain string.
    """
    n_lower_brackets = domain_string.count("[")
    n_lower_parenthesis = domain_string.count("(")
    n_lower = n_lower_brackets + n_lower_parenthesis
    n_upper_brackets = domain_string.count("]")
    n_upper_parenthesis = domain_string.count(")")
    n_upper = n_upper_brackets + n_upper_parenthesis
    if n_lower != n_upper:
        raise RuntimeError(f"The given domain string '{domain_string}' is invalid!")
    return n_lower


def vectorize_numpy_dict(numpy_dict: dict) -> np.ndarray:
    """
    Concatenates all vectors from a dict to a single vector and returns it. For example
    when given the dict {'y1': np.array([1, 2, 3]), 'y2': np.array([4, 5, 6])} it will
    be returned np.array([1, 2, 3, 4, 5, 6]).

    Parameters
    ----------
    numpy_dict
        The keys are usually strings stating some sensor name, while the values are 1D
        numpy-arrays, either from a forward model response or experimental data.

    Returns
    -------
    vector
        Contains the numeric data contained in numpy-dict in a single vector.
    """
    n_list = []
    for numpy_vector in numpy_dict.values():
        n_list.append(np.size(numpy_vector))
    n = sum(n_list)
    vector = np.zeros(n)
    idx_start = 0
    for numpy_vector, n_i in zip(numpy_dict.values(), n_list):
        idx_end = idx_start + n_i
        vector[idx_start:idx_end] = numpy_vector
        idx_start = idx_end
    return vector


def assemble_covariance_matrix(
    coords_array: np.ndarray,
    std_model: Union[int, float, np.ndarray],
    std_meas: Union[int, float, np.ndarray, None],
    l_corr: Union[int, float],
    y_model: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Assembles and returns the full covariance matrix.

    Parameters
    ----------
    coords_array
        Has shape (n, d) where n is the number of points and d is the number of
        spatial variables. Each row is a coordinate vector of a single point.
    std_model
        The vector of model error standard deviations.
    std_meas
        The vector of measurement error standard deviations, if this type of error
        is being considered. Otherwise None.
    l_corr
        The spatial correlation length.
    y_model
        The model response. Needed only for multiplicative models. Otherwise None.
        If this value is not None, a multiplicative model is assumed.

    Returns
    -------
    cov_matrix
         The full covariance matrix in form of an (n, n)-array.
    """

    # assemble the covariance matrix for additive model error only
    f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
    std1, std2 = np.meshgrid(std_model, std_model)
    cov_matrix = std1 * std2 * correlation_matrix(coords_array, f_corr)

    # adjust the covariance matrix for multiplicative model error
    if y_model is not None:
        y1, y2 = np.meshgrid(y_model, y_model)
        cov_matrix = y1 * y2 * cov_matrix

    # adjust the covariance matrix if an additive measurement error is considered
    if std_meas is not None:
        n = len_or_one(coords_array)
        cov_matrix += std_meas**2 * np.eye(n)

    return cov_matrix


def get_shape_2d(array_or_tuple: Union[np.ndarray, tuple]) -> Tuple[int, int]:
    """
    Returns the number of rows and columns from a one or two dimensional numpy array or
    tuple.

    Parameters
    ----------
    array_or_tuple
        The given array of one or two dimensions.

    Returns
    -------
        number_of_rows
            The number of rows found in the array.
        number_of_cols
            The number of columns found in the array.
    """

    # convert tuple to array if a tuple is given
    if isinstance(array_or_tuple, tuple):
        array = np.array(array_or_tuple)
    else:
        array = array_or_tuple

    nd = len(array.shape)
    if nd == 1:
        number_of_rows = array.shape[0]
        number_of_cols = 1
    elif nd == 2:
        number_of_rows = array.shape[0]
        number_of_cols = array.shape[1]
    else:
        raise ValueError(
            f"The function get_shape_2D is not intended for arrays with more than 2 "
            f"dimensions. However, the given array has {nd} dimensions."
        )
    return number_of_rows, number_of_cols


def convert_to_tuple(a: Union[int, float, np.ndarray]) -> Union[int, float, tuple]:
    """
    Converts a given numpy array into a respective tuple.

    Parameters
    ----------
    a
        The given numpy array. Might also be a scalar (float/int) which will be
        returned unprocessed.

    Returns
    -------
        The given array as a tuple.
    """

    if isinstance(a, np.ndarray):
        if len(a.shape) == 1:
            return tuple(a)
        elif len(a.shape) == 2:
            return tuple(tuple(row) for row in a)
        else:
            raise ValueError(
                f"Arrays of higher dimension than 2 are not supported! (The given "
                f"array was {a})"
            )
    else:
        return a


def safe_string(string: str, n_max: int = 255) -> str:
    """
    Converts a given string to a string that can be used to name a file by removing
    non-alphanumeric characters and some other checks and operations.

    Parameter
    ---------
    string
        The given string.
    n_max
        The maximum number of characters the returned string is allowed to have.

    Returns
    -------
    safe
        The cleaned string that can be safely used for file-naming.
    """

    # clean the string
    safe = ""
    for char in string:
        if char.isalnum():
            safe += char
        elif char == "_":
            safe += char
        elif char.isspace():
            safe += "_"

    # the last character should not be an underscore
    if safe.endswith("_"):
        safe = safe[:-1]

    # a filename cannot start with a digit
    if safe[0].isdigit():
        safe = "_" + safe

    # crop name if it is too long
    if len(safe) > n_max:
        safe = safe[:n_max]

    return safe


def synchronize_objects(
    new_obj: object, ref_obj: object, exclude_startswith: tuple = ("_", "__")
):
    """
    Copies all attributes from a reference object (ref_obj) to a given object (new_obj)
    if those attributes exist in both objects. For example, if new_obj.a = None and
    ref_obj.a = 3 then this function will result in new_obj.a = 3. Excluded from this
    synchronization are properties and functions.

    Parameters
    ----------
    new_obj
        The new object that should get the attribute-values of ref_obj.
    ref_obj
        The reference object that should 'give' its attributes to new_obj.
    exclude_startswith
        All attributes that start with one of the strings given in this tuple will not
        be copied.
    """
    for attribute in dir(ref_obj):
        attr_is_valid = True
        for s in exclude_startswith:
            if attribute.startswith(s):
                attr_is_valid = False
                break
        # make sure that the attribute is not a property
        if isinstance(getattr(type(ref_obj), attribute, None), property):
            attr_is_valid = False
        # make sure that the attribute is not a function
        if isinstance(getattr(type(ref_obj), attribute, None), types.FunctionType):
            attr_is_valid = False
        if attr_is_valid:
            if hasattr(new_obj, attribute):
                try:
                    setattr(new_obj, attribute, deepcopy(getattr(ref_obj, attribute)))
                except AttributeError:
                    raise AttributeError(f"can't set attribute '{attribute}'")


class HiddenPrints:
    """
    Allows to create a context manager that suppresses prints to stdout. Taken from
    stackoverflow, apparently originally proposed by @FakeRainBrigand. In probeye, this
    class is currently only used to prevent some outputs from other packages during the
    generation of the documentation.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
