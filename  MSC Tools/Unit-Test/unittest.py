import unittest

def func(x):
    return x^2

class func_test(unittest.TestCase):
    """Test for negative,
     test for float,
      test for zero,
       test for large"""
    def test(self):
        self.assertEqual(func(0), 0)
        self.assertEqual(func(10), 100)
        self.assertEqual(func(-10), 100)

if __name__ == __main__:
    func_test(func).test()
    print(func(12))
