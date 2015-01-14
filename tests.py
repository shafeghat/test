"""
Unit tests.
"""
import unittest

import summator



class TestSummator(unittest.TestCase):
    """
    Tests the `summator` module by checking its results against the first
    10 expected results.
    """    

    def setUp(self):
        """
        Initialise an array containing the expected results.
        """
        self.first_n = 10
        self.first_sums = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    
    
    def test_first_sums(self):
        """
        Checking the first 10 expected results match those generated.
        """
        for i in range(self.first_n):
            expected = self.first_sums[i]
            actual = summator.sum_n(i)
            self.assertEqual(expected, actual)
        
        

if __name__ == '__main__':
    unittest.main()
