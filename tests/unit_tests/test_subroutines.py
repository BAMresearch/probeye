# third party imports
import unittest

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import delta_x
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

    def test_delta_x(self):
        # the returned step size (intended for a finite difference scheme) must
        # always be positive (greater than zero)
        self.assertGreater(delta_x(-1.0), 0)
        self.assertGreater(delta_x(0.0), 0)
        self.assertGreater(delta_x(1.0), 0)
        self.assertEqual(delta_x(0, delta=1), 1)

    def test_list2dict(self):
        # check standard cases
        self.assertEqual(list2dict(['a', 'b']),
                         {'a': 'a', 'b': 'b'})
        self.assertEqual(list2dict([{'a': 'm'}, 'b']),
                         {'a': 'm', 'b': 'b'})
        self.assertEqual(list2dict([{'a': 'm'}, {'b': 'c'}]),
                         {'a': 'm', 'b': 'c'})
        # the input cannot be a dictionary
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            list2dict({'a': 'm', 'b': 'c'})
        # only one-element dictionaries are valid
        with self.assertRaises(ValueError):
            list2dict([{'a': 'm', 'b': 'c'}])
        # the input cannot contain numbers
        with self.assertRaises(TypeError):
            list2dict(['a', 1.0])

if __name__ == "__main__":
    unittest.main()
