# standard library imports
from loguru import logger
import unittest
import os

# third party imports
import numpy as np

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.subroutines import len_or_one
from probeye.subroutines import make_list
from probeye.subroutines import underlined_string
from probeye.subroutines import titled_table
from probeye.subroutines import replace_string_chars
from probeye.subroutines import simplified_list_string
from probeye.subroutines import simplified_dict_string
from probeye.subroutines import sub_when_empty
from probeye.subroutines import dict2list
from probeye.subroutines import list2dict
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import flatten
from probeye.subroutines import process_spatiotemporal_coordinates
from probeye.subroutines import translate_prms_def
from probeye.subroutines import print_probeye_header
from probeye.subroutines import logging_setup
from probeye.subroutines import print_dict_in_rows
from probeye.subroutines import add_index_to_tex_prm_name
from probeye.subroutines import check_for_uninformative_priors
from probeye.subroutines import compute_reduction_array
from probeye.subroutines import get_dictionary_depth
from probeye.subroutines import incrementalize
from probeye.subroutines import extract_true_values
from probeye.subroutines import translate_simple_correlation
from probeye.subroutines import get_global_name
from probeye.subroutines import count_intervals
from probeye.subroutines import vectorize_numpy_dict
from probeye.subroutines import HiddenPrints


class TestProblem(unittest.TestCase):
    def test_len_or_one(self):
        # check for objects with no len-method
        self.assertEqual(len_or_one(1), 1)
        self.assertEqual(len_or_one(1.0), 1)
        self.assertEqual(len_or_one(True), 1)
        self.assertEqual(len_or_one("a"), 1)
        self.assertEqual(len_or_one(None), 1)
        # check for lists with zero or more elements
        self.assertEqual(len_or_one([]), 0)
        self.assertEqual(len_or_one([1]), 1)
        self.assertEqual(len_or_one([1, 2]), 2)
        self.assertEqual(len_or_one([1, 2, 3]), 3)
        # check for the weird numpy case (the object np.array(1.0) does have len-method,
        # but when you do len(np.array(1.0)) you get an error)
        self.assertEqual(len_or_one(np.array(1.0)), 1)

    def test_make_list(self):
        # check main use for single non-list input
        self.assertEqual(make_list("a"), ["a"])
        # check common use for lists
        self.assertEqual(make_list(["a"]), ["a"])
        self.assertEqual(make_list(["a", "b"]), ["a", "b"])
        self.assertEqual(make_list([]), [])
        # check common use for tuples
        self.assertEqual(make_list(("a", "b")), ("a", "b"))
        self.assertEqual(make_list(()), ())

    def test_underlined_string(self):
        # check common use cases
        self.assertEqual(underlined_string("Test"), "\nTest\n====\n")
        self.assertEqual(
            underlined_string("ABC", n_empty_start=0, n_empty_end=2, symbol="-"),
            "ABC\n---\n\n",
        )

    def test_titled_table(self):
        # check common use case
        titled_table_str = titled_table(
            "Title", "Col Col", symbol="=", n_empty_start=2, n_empty_end=1
        )
        self.assertEqual(titled_table_str, "\n\nTitle\n=======\nCol Col\n\n")

    def test_replace_string_chars(self):
        # check common use case
        self.assertEqual(
            replace_string_chars("Test1233", replace={"T": "t", "3": "4"}), "test1244"
        )
        self.assertEqual(replace_string_chars("Test1233", remove=["1", "e"]), "Tst233")
        self.assertEqual(
            replace_string_chars("Test1233", replace={"3": "4"}, remove=["1"]),
            "Test244",
        )

    def test_simplified_list_string(self):
        # check common use case
        self.assertEqual(simplified_list_string(["a", "b"]), "a, b")
        self.assertEqual(simplified_list_string(["a"]), "a")

    def test_simplified_dict_string(self):
        # check common use case
        self.assertEqual(simplified_dict_string({"a": 1, "b": 2}), "a=1, b=2")
        self.assertEqual(simplified_dict_string({"a": 1.0}), "a=1.0")

    def test_sub_when_empty(self):
        # check common use cases
        self.assertEqual(sub_when_empty("test"), "test")
        self.assertEqual(sub_when_empty("", empty_str="-"), "-")
        # check for cases with invalid input
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            sub_when_empty(1.0)

    def test_dict2list(self):
        # check common use cases
        self.assertEqual(dict2list({"a": 1, "b": 2}), [{"a": 1}, {"b": 2}])
        self.assertEqual(dict2list({"a": 1}), [{"a": 1}])
        # check for cases with invalid input
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            dict2list([{"a": 1}, {"b": 2}])

    def test_list2dict(self):
        # check common use cases
        self.assertEqual(list2dict(["a", "b"]), {"a": "a", "b": "b"})
        self.assertEqual(list2dict([{"a": "m"}, "b"]), {"a": "m", "b": "b"})
        self.assertEqual(list2dict([{"a": "m"}, {"b": "c"}]), {"a": "m", "b": "c"})
        self.assertEqual(list2dict({"a": "m", "b": "c"}), {"a": "m", "b": "c"})
        # each dict in the list must not contain more than one element
        with self.assertRaises(ValueError):
            list2dict([{"a": "m1", "b": "b"}])
        # the input cannot contain numbers
        with self.assertRaises(TypeError):
            list2dict(["a", 1.0])
        # the input must be of type list or dict
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            list2dict((1, 2))

    def test_pretty_time_delta(self):
        # check the different output formats
        self.assertEqual(pretty_time_delta(1), "1s")
        self.assertEqual(pretty_time_delta(62), "1m2s")
        self.assertEqual(pretty_time_delta(3723), "1h2m3s")
        self.assertEqual(pretty_time_delta(93780), "1d2h3m")

    def test_flatten(self):
        # test for a simple nested list
        nested_list_1 = [1, 2, [3, 4]]
        computed_result = flatten(nested_list_1)
        expected_result = [1, 2, 3, 4]
        self.assertEqual(computed_result, expected_result)
        # test for another simple nested list
        nested_list_2 = [[1, 2, [3, 4]]]
        computed_result = flatten(nested_list_2)
        expected_result = [1, 2, 3, 4]
        self.assertEqual(computed_result, expected_result)
        # test for a flat list
        flat_list = [1, 2, 3, 4]
        computed_result = flatten(flat_list)
        expected_result = [1, 2, 3, 4]
        self.assertEqual(computed_result, expected_result)
        # test for a simple nested numpy-array
        nested_array_1 = np.array([[1, 2], [3, 4]])
        computed_result = flatten(nested_array_1)
        expected_result = [1, 2, 3, 4]
        self.assertEqual(computed_result, expected_result)
        # test for another simple nested numpy-array
        nested_array_2 = np.array([[1, 2, 3, 4]])
        computed_result = flatten(nested_array_2)
        expected_result = [1, 2, 3, 4]
        self.assertEqual(computed_result, expected_result)
        # test for a flat numpy-array
        flat_array = np.array([1, 2, 3, 4])
        computed_result = flatten(flat_array)
        expected_result = [1, 2, 3, 4]
        self.assertEqual(computed_result, expected_result)
        # test for use with scalar
        scalar = 1.2
        computed_result = flatten(scalar)
        expected_result = [1.2]
        self.assertEqual(computed_result, expected_result)
        # test for the None-input
        none_arg = None
        computed_result = flatten(none_arg)
        expected_result = None
        self.assertEqual(computed_result, expected_result)
        # check invalid input (here: invalid type)
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            flatten((1, 2, 3))

    def test_process_spatial_coordinates(self):
        # check for complete None-input
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=None, z=None, coords=None
        )
        coords_expected = np.array([])
        order_expected = []
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for only x being given as numpy array
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=np.array([1, 2, 3]), y=None, z=None, coords=None
        )
        coords_expected = np.array([[1, 2, 3]])
        order_expected = ["x"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for only x being given as list
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=[1, 2, 3], y=None, z=None, coords=None
        )
        coords_expected = np.array([[1, 2, 3]])
        order_expected = ["x"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for only y being given as numpy array
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=np.array([1, 2, 3]), z=None, coords=None
        )
        coords_expected = np.array([[1, 2, 3]])
        order_expected = ["y"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for only y being given as list
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=[1, 2, 3], z=None, coords=None
        )
        coords_expected = np.array([[1, 2, 3]])
        order_expected = ["y"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for only z being given as numpy array
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=None, z=np.array([1, 2, 3]), coords=None
        )
        coords_expected = np.array([[1, 2, 3]])
        order_expected = ["z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for only z being given as list
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=None, z=[1, 2, 3], coords=None
        )
        coords_expected = np.array([[1, 2, 3]])
        order_expected = ["z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for x and y being given as numpy arrays
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=np.array([1, 2, 3]), y=np.array([4, 5, 6]), z=None, coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6]])
        order_expected = ["x", "y"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for x and y being given as lists
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=[1, 2, 3], y=[4, 5, 6], z=None, coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6]])
        order_expected = ["x", "y"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for x and z being given as numpy arrays
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=np.array([1, 2, 3]), y=None, z=np.array([4, 5, 6]), coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6]])
        order_expected = ["x", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for x and z being given as lists
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=[1, 2, 3], y=None, z=[4, 5, 6], coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6]])
        order_expected = ["x", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for y and z being given as numpy arrays
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=np.array([1, 2, 3]), z=np.array([4, 5, 6]), coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6]])
        order_expected = ["y", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for y and z being given as lists
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=None, y=[1, 2, 3], z=[4, 5, 6], coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6]])
        order_expected = ["y", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for x, y and z being given as numpy arrays
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            z=np.array([7, 8, 9]),
            coords=None,
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        order_expected = ["x", "y", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for y and z being given as lists
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9], coords=None
        )
        coords_expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        order_expected = ["x", "y", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check for coords being given as numpy array with new order
        coords_computed, order_computed = process_spatiotemporal_coordinates(
            coords=np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]), order=("y", "x", "z")
        )
        coords_expected = np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]])
        order_expected = ["y", "x", "z"]
        self.assertTrue(
            np.allclose(coords_computed, coords_expected)
            and coords_computed.shape == coords_expected.shape
        )
        self.assertEqual(order_computed, order_expected)
        # check invalid input (here: invalid type)
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            process_spatiotemporal_coordinates(coords=(1, 2, 3, 4, 5))
        # check invalid input (here: inconsistent individual lengths)
        with self.assertRaises(RuntimeError):
            process_spatiotemporal_coordinates(x=[1], y=[1, 2])

    def test_translate_prms_def(self):
        # valid use case: single string
        prms_def_given = "sigma"
        expected_result = {"sigma": "sigma"}
        computed_result, _ = translate_prms_def(prms_def_given)
        self.assertEqual(computed_result, expected_result)
        # valid use case: single string as list
        prms_def_given = ["sigma"]
        expected_result = {"sigma": "sigma"}
        computed_result, _ = translate_prms_def(prms_def_given)
        self.assertEqual(computed_result, expected_result)
        # valid use case: single 1-element dict
        prms_def_given = {"sigma": "sigma"}
        expected_result = {"sigma": "sigma"}
        computed_result, _ = translate_prms_def(prms_def_given)
        self.assertEqual(computed_result, expected_result)
        # valid use case: multiple-element dict
        prms_def_given = {"sigma": "std", "mu": "mean"}
        expected_result = {"sigma": "std", "mu": "mean"}
        computed_result, _ = translate_prms_def(prms_def_given)
        self.assertEqual(computed_result, expected_result)
        # valid use case: multiple-element list of strings
        prms_def_given = ["sigma", "mu"]
        expected_result = {"sigma": "sigma", "mu": "mu"}
        computed_result, _ = translate_prms_def(prms_def_given)
        self.assertEqual(computed_result, expected_result)
        # valid use case: mixed multiple-element list
        prms_def_given = ["sigma", {"mu": "mean"}]
        expected_result = {"sigma": "sigma", "mu": "mean"}
        computed_result, _ = translate_prms_def(prms_def_given)
        self.assertEqual(computed_result, expected_result)
        # invalid use case: list with 2-element dict
        with self.assertRaises(ValueError):
            translate_prms_def([{"sigma": "sigma", "mu": "mean"}])

    def test_print_probeye_header(self):
        # simply check that the defaults work
        print_probeye_header(use_logger=True)
        print_probeye_header(use_logger=False)

    def test_logging_setup(self):
        # check that the log-file option works
        log_dir = os.path.dirname(__file__)
        log_file = os.path.join(log_dir, "logfile.txt")
        logging_setup(log_file=log_file)
        logger.info("This text should also be written to the log-file.")
        self.assertTrue(os.path.exists(log_file))
        # now, check the overwrite-option
        logger.remove()  # necessary to remove the logfile
        logging_setup(log_file=log_file, overwrite_log_file=True)
        logger.info("This text should be overwriting the previous one.")
        self.assertTrue(os.path.exists(log_file))
        logger.remove()  # necessary to remove the logfile
        os.remove(log_file)

    def test_print_dict_in_rows(self):
        # this check covers the val_fmt options
        print_dict_in_rows({"a": 1.2345, "b": 3.45365}, val_fmt=".2f")

    def test_add_index_to_tex_prm_name(self):
        # check normal use case 1
        computed_result = add_index_to_tex_prm_name(r"$\alpha$", 1)
        expected_result = r"$\alpha_1$"
        self.assertEqual(computed_result, expected_result)
        # check normal use case 2
        computed_result = add_index_to_tex_prm_name("$\\alpha$", 2)
        expected_result = "$\\alpha_2$"
        self.assertEqual(computed_result, expected_result)
        # check normal use case 3
        computed_result = add_index_to_tex_prm_name("$a$", 3)
        expected_result = "$a_3$"
        self.assertEqual(computed_result, expected_result)
        # check normal use case 4 (note the missing '$'-signs)
        computed_result = add_index_to_tex_prm_name("a", 4)
        expected_result = "a (4)"
        self.assertEqual(computed_result, expected_result)
        # check normal use case 5
        computed_result = add_index_to_tex_prm_name("$a_c$", 5)
        expected_result = "$a_c$ (5)"
        self.assertEqual(computed_result, expected_result)
        # check normal use case 6
        computed_result = add_index_to_tex_prm_name("$\\gamma^c$", 6)
        expected_result = "$\\gamma^c$ (6)"
        self.assertEqual(computed_result, expected_result)

    def test_check_for_uninformative_priors(self):
        # check if the detection works as expected
        problem = InverseProblem("Problem with uninformative prior")
        problem.add_parameter(
            "sigma", "likelihood", prior=("uniform", {"low": 0.1, "high": 0.8})
        )
        check_for_uninformative_priors(problem)  # no error should be raised
        problem.add_parameter("m", "model")  # uninformative prior
        with self.assertRaises(RuntimeError):
            check_for_uninformative_priors(problem)

    def test_get_dictionary_depth(self):
        # check for a depth-1-dictionary
        compute_result = get_dictionary_depth({"a": None})
        expected_result = 1
        self.assertEqual(compute_result, expected_result)
        # check for a depth-2-dictionary
        compute_result = get_dictionary_depth({"a": {"b": None}})
        expected_result = 2
        self.assertEqual(compute_result, expected_result)

    def test_compute_reduction_array(self):
        # check for a simple example
        array = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        red_array_computed, rows_to_remove_computed = compute_reduction_array(array)
        red_array_expected = np.array([[0.5, 0.5, 0], [0, 0, 1]])
        rows_to_remove_expected = [1]
        self.assertTrue(np.allclose(red_array_computed, red_array_expected))
        self.assertEqual(rows_to_remove_computed, rows_to_remove_expected)

    def test_incrementalize(self):

        # no incrementalization is required
        v_in = [1, 2, 3, 4]
        v_out, f, flag = incrementalize(v_in)
        self.assertTrue(np.allclose(v_in, v_out))
        self.assertTrue(np.allclose(v_in, f(v_in)))
        self.assertTrue(flag)

        # same test as before, but with np.array instead of list
        v_in = np.array([1, 2, 3, 4])
        v_out, f, flag = incrementalize(v_in)
        self.assertTrue(np.allclose(v_in, v_out))
        self.assertTrue(np.allclose(v_in, f(v_in)))
        self.assertTrue(flag)

        # here, the input vector is not sorted and has no duplicate elements
        v_in = np.array([3, 1, 4, 2])
        v_out, f, flag = incrementalize(v_in)
        v_out_expected = np.array([1, 2, 3, 4])
        self.assertTrue(np.allclose(v_out_expected, v_out))
        self.assertTrue(np.allclose(v_out_expected, f(v_in)))
        self.assertTrue(not flag)

        # here, the input vector is not sorted and has duplicate elements
        v_in = np.array([3, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2])
        v_out, f, flag = incrementalize(v_in)
        v_out_expected = np.array([1, 2, 3, 4])
        self.assertTrue(np.allclose(v_out_expected, v_out))
        self.assertTrue(np.allclose(v_out_expected, f(v_in)))
        self.assertTrue(not flag)

        # special case: input has only one element
        v_in = [1]
        v_out, f, flag = incrementalize(v_in)
        self.assertTrue(np.allclose(v_in, v_out))
        self.assertTrue(np.allclose(v_in, f(v_in)))
        self.assertTrue(flag)

    def test_extract_true_values(self):

        # this is the default use case
        true_values = {"a": 1, "b": 2, "c": 3}
        var_names = ["b", "a", "c"]
        expected_array = np.array([2, 1, 3])
        computed_array = extract_true_values(true_values, var_names)
        self.assertTrue(np.allclose(expected_array, computed_array))

        # this is the use case for a vector-valued parameter
        true_values = {"a": 1, "mb": np.array([2, 3]), "c": 4}
        var_names = ["mb_1", "mb_2", "c", "a"]
        expected_array = np.array([2, 3, 4, 1])
        computed_array = extract_true_values(true_values, var_names)
        self.assertTrue(np.allclose(expected_array, computed_array))

        # same as before, but with list instead of array and more underscores
        true_values = {"a": 1, "m_and_b": [2, 3], "c": 4}
        var_names = ["m_and_b_1", "m_and_b_2", "c", "a"]
        expected_array = np.array([2, 3, 4, 1])
        computed_array = extract_true_values(true_values, var_names)
        self.assertTrue(np.allclose(expected_array, computed_array))

    def test_translate_simple_correlation(self):

        # standard use case
        computed_value = translate_simple_correlation("T1:xy")
        expected_value = {"T1": {"x": "x", "y": "y"}}
        self.assertEqual(computed_value, expected_value)

        # invalid input with a non-standard correlation variable
        with self.assertRaises(ValueError):
            translate_simple_correlation("T1:xu")

        # invalid input with no colon
        with self.assertRaises(ValueError):
            translate_simple_correlation("T1xy")

        # invalid input with more than one colon
        with self.assertRaises(ValueError):
            translate_simple_correlation("T1:x:y")

    def test_get_global_name(self):

        # default use case
        computed_value = get_global_name("a_loc", {"a_glob": "a_loc", "b": "B"})
        expected_value = "a_glob"
        self.assertEqual(computed_value, expected_value)

        # case when the given local name is not found
        with self.assertRaises(RuntimeError):
            get_global_name("not_a_value", {"a_glob": "a_loc", "b": "B"})

    def test_count_intervals(self):

        # default use case
        computed_value = count_intervals("[0, 1](0,1)(0,1][0, 1)")
        expected_value = 4
        self.assertEqual(computed_value, expected_value)

        # invalid input with an incomplete domain
        with self.assertRaises(RuntimeError):
            count_intervals("[0, 1](0,1)(0,1][0, 1")

    def test_vectorize_numpy_dict(self):

        # default use case
        d = {"y1": np.array([1.0, 2.0, 3.0])}
        computed_value = vectorize_numpy_dict(d)
        expected_value = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(computed_value, expected_value))

        # default use case
        d = {"y1": np.array([1.0, 2.0, 3.0]), "y2": np.array([4.0, 5.0, 6.0])}
        computed_value = vectorize_numpy_dict(d)
        expected_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertTrue(np.allclose(computed_value, expected_value))

    def test_HiddenPrints(self):

        # default use case for coverage
        with HiddenPrints():
            print("This string will not appear in stdout.")


if __name__ == "__main__":
    unittest.main()
