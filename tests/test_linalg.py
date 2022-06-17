import unittest
from mathtypes.linalg import matrix, vector


class TestVector(unittest.TestCase):

    def test_append(self):
        x = vector([1, 3, 1])
        with self.assertRaises(TypeError):
            x.append('a')
        x.append(-1)
        self.assertListEqual(list(x), [1, 3, 1, -1])

    def test_norm(self):
        x = vector()
        self.assertEqual(x.norm(), 0)

        x = vector([5, 4, 6, -2])
        self.assertEqual(x.norm(), 9)

        x = vector([1, 2.5, -4, 1])
        self.assertAlmostEqual(x.norm(), 4.9244289008980523608731057074588)

    def test_insert(self):
        x = vector([1, 1])
        with self.assertRaises(TypeError):
            x.insert(0, 'a')

        x.insert(1, 2)
        self.assertEqual(x[1], 2)

        x.insert(-1, 4)
        self.assertEqual(x[-2], 4)

    def test_add(self):
        a = vector()
        b = vector()
        self.assertEqual(list(a + b), [])

        a = vector([1, 1, 1])
        b = vector([1, 1])
        with self.assertRaises(IndexError):
            a + b

        b = vector([-2, -3, -1.5])
        self.assertListEqual(list(a + b), [-1, -2, -0.5])

    def test_sub(self):
        a = vector()
        b = vector()
        self.assertEqual(list(a - b), [])

        a = vector([1, 1, 1])
        b = vector([1, 1])
        with self.assertRaises(IndexError):
            a - b

        b = vector([-2, -3, 1.5])
        self.assertListEqual(list(a - b), [3, 4, -0.5])

    def test_rmul(self):
        a = vector()
        self.assertEqual(list(2 * a), [])

        a = vector([1, 2, -1])
        with self.assertRaises(TypeError):
            "a" * a

        self.assertListEqual(list(4 * a), [4, 8, -4])


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.A = matrix([[1, 3, 2], [3, -2, -0.5], [1, 0, 1]])

    def test_insert(self):
        values1 = [1, -2, 4, 11]
        with self.assertRaisesRegex(ValueError,
                                    "The size of the row to be inserted is greater than the size of the matrix rows"):
            self.A.insert(0, 0, values1)

        with self.assertRaisesRegex(ValueError,
                                    "The size of column to be inserted is greater than the size of the matrix columns"):
            self.A.insert(1, self.A.cols, values1)

        values2 = [1, -2, 4]
        self.A.insert(0, 0, values2)
        self.assertListEqual(
            list(self.A), [[1, -2, 4], [1, 3, 2], [3, -2, -0.5], [1, 0, 1]])

        self.A.insert(1, self.A.cols, values1)
        self.assertListEqual(list(self.A),
                             [[1, -2, 4, 1], [1, 3, 2, -2], [3, -2, -0.5, 4], [1, 0, 1, 11]])

    def test_multiply_vector(self):
        b = 1
        with self.assertRaises(TypeError):
            self.A.multiply(b)

        b = vector([1, 1])
        with self.assertRaises(IndexError):
            self.A.multiply(b)

        b.append(-1)
        self.assertListEqual(list(self.A.multiply(b)), [2, 1.5, 0])

    def test_multiply_matrix(self):
        b = matrix([[1, 1, 1], [-1, 2, 1]])
        with self.assertRaises(IndexError):
            self.A.multiply(b)

        b = matrix([[1, 1], [-1, 2], [-1, 1]])
        self.assertListEqual(list(self.A.multiply(b)), [
                             [-4, 9], [5.5, -1.5], [0, 2]])

    def test_transpose(self):
        self.assertListEqual(list(self.A.transpose()), [
                             [1, 3, 1], [3, -2, 0], [2, -0.5, 1]])

    def test_inverse(self):
        A = matrix([[1, 9], [0, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "Not a square matrix"):
            A.inverse()

        A = matrix([[1, 0, 1], [-2, 2, 3], [-1, 0, -1]])
        with self.assertRaisesRegex(ValueError, expected_regex="Singular matrix"):
            A.inverse()

        A[2, 2] = 3
        self.assertListEqual(list(A.inverse()), [
                             [0.75, 0.0, -0.25], [0.375, 0.5, -0.625], [0.25, 0.0, 0.25]])
