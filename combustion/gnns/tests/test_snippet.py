"""
Snippet for setting up tests
"""

import unittest


class Testing(unittest.TestCase):
    """
    Testing class
    """

    def test_string(self):
        """
        test_string
        """
        a_func_1 = 'some'
        b_func_1 = 'some'
        self.assertEqual(a_func_1, b_func_1)

    def test_boolean(self):
        """
        test_boolean
        """
        a_func_2 = True
        b_func_2 = True
        self.assertEqual(a_func_2, b_func_2)

if __name__ == '__main__':
    unittest.main()
