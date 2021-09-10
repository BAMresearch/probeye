# third party imports
import unittest

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import make_list
from probeye.subroutines import underlined_string
from probeye.subroutines import titled_table
from probeye.subroutines import replace_string_chars
from probeye.subroutines import simplified_list_string
from probeye.subroutines import simplified_dict_string
from probeye.subroutines import unvectorize_dict_values
from probeye.subroutines import sub_when_empty
from probeye.subroutines import dict2list
from probeye.subroutines import list2dict

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

    def test_make_list(self):
        # check main use for single non-list input
        self.assertEqual(make_list('a'), ['a'])
        # check common use for lists
        self.assertEqual(make_list(['a']), ['a'])
        self.assertEqual(make_list(['a', 'b']), ['a', 'b'])
        self.assertEqual(make_list([]), [])
        # check common use for tuples
        self.assertEqual(make_list(('a', 'b')), ('a', 'b'))
        self.assertEqual(make_list(()), ())

    def test_underlined_string(self):
        # check common use cases
        self.assertEqual(underlined_string("Test"), "\nTest\n════\n")
        self.assertEqual(underlined_string(
            "ABC", n_empty_start=0, n_empty_end=2, symbol="-"), "ABC\n---\n\n")

    def test_titled_table(self):
        # check common use case
        titled_table_str = titled_table(
            'Title', 'Col Col', symbol='=', n_empty_start=2, n_empty_end=1)
        self.assertEqual(titled_table_str, "\n\nTitle\n=======\nCol Col\n\n")

    def test_replace_string_chars(self):
        # check common use case
        self.assertEqual(replace_string_chars(
            'Test1233', replace={'T': 't', '3': '4'}), 'test1244')
        self.assertEqual(replace_string_chars(
            'Test1233', remove=['1', 'e']), 'Tst233')
        self.assertEqual(replace_string_chars(
            'Test1233', replace={'3': '4'}, remove=['1']), 'Test244')

    def test_simplified_list_string(self):
        # check common use case
        self.assertEqual(simplified_list_string(['a', 'b']), 'a, b')
        self.assertEqual(simplified_list_string(['a']), 'a')

    def test_simplified_dict_string(self):
        # check common use case
        self.assertEqual(simplified_dict_string({'a': 1, 'b': 2}), 'a=1, b=2')
        self.assertEqual(simplified_dict_string({'a': 1.0}), 'a=1.0')

    def test_unvectorize_dict_values(self):
        # check common use case
        self.assertEqual(unvectorize_dict_values({'x': [1, 2], 'y': [3, 4]}),
                         [{'x': 1, 'y': 3}, {'x': 2, 'y': 4}])
        # check for cases with invalid input
        with self.assertRaises(RuntimeError):
            # values cannot have different lengths
            unvectorize_dict_values({'x': [1, 2, 3], 'y': [3, 4]})

    def test_sub_when_empty(self):
        # check common use cases
        self.assertEqual(sub_when_empty('test'), 'test')
        self.assertEqual(sub_when_empty('', empty_str='-'), '-')
        # check for cases with invalid input
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            sub_when_empty(1.0)

    def test_dict2list(self):
        # check common use cases
        self.assertEqual(dict2list({'a': 1, 'b': 2}), [{'a': 1}, {'b': 2}])
        self.assertEqual(dict2list({'a': 1}), [{'a': 1}])
        # check for cases with invalid input
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            dict2list([{'a': 1}, {'b': 2}])

    def test_list2dict(self):
        # check common use cases
        self.assertEqual(list2dict(['a', 'b']),
                         {'a': 'a', 'b': 'b'})
        self.assertEqual(list2dict([{'a': 'm'}, 'b']),
                         {'a': 'm', 'b': 'b'})
        self.assertEqual(list2dict([{'a': 'm'}, {'b': 'c'}]),
                         {'a': 'm', 'b': 'c'})
        self.assertEqual(list2dict({'a': 'm', 'b': 'c'}),
                         {'a': 'm', 'b': 'c'})
        # each dict in the list must not contain more than one element
        with self.assertRaises(ValueError):
            list2dict([{'a': 'm1', 'b': 'b'}])
        # the input cannot contain numbers
        with self.assertRaises(TypeError):
            list2dict(['a', 1.0])

if __name__ == "__main__":
    unittest.main()
