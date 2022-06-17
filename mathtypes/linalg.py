from ast import Index
from typing import Generic, MutableSequence, Union, overload, SupportsIndex, TypeVar
from math import sqrt
import copy


T = TypeVar('T', int, float)


class vector(MutableSequence[T], Generic[T]):

    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, _values: list[T]):
        ...

    @overload
    def __init__(self, n: int):
        ...

    def __init__(self, arg: Union[list[T], int] = None) -> None:
        if isinstance(arg, list):
            if all(isinstance(val, (int, float)) for val in arg):
                self._values = arg.copy()
            else:
                raise TypeError("list items must be integer or floating point")
        elif arg is not None and isinstance(arg, int):
            self._values = [0] * arg
        else:
            self._values = []
        self.length = len(self._values)

    def __getitem__(self, i: SupportsIndex) -> T:
        return self._values[i]

    def __setitem__(self, i: SupportsIndex, value: T):
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be an integer or float")
        self._values[i] = value

    def __delitem__(self, i: SupportsIndex):
        del self._values[i]

    def __add__(self, other: "vector[T]") -> "vector[T]":
        if self.length == other.length:
            return vector([self._values[i] + other[i] for i in range(self.length)])
        raise IndexError(f'expected vector with length {self.length}')

    def __sub__(self, other: "vector[T]") -> "vector[T]":
        if self.length == other.length:
            return vector([self._values[i] - other[i] for i in range(self.length)])
        raise IndexError(f'expected vector with length {self.length}')

    def __neg__(self) -> "vector[T]":
        return vector([-val for val in self._values])

    def __rmul__(self, factor: T) -> "vector[T]":
        if isinstance(factor, (int, float)):
            return vector([value * factor for value in self._values])
        raise TypeError("factor must be an integer or a float")

    def __eq__(self, other: "vector[T]") -> bool:
        if isinstance(other, vector) and self.length == other.length:
            for i in range(self.length):
                if self[i] != other[i]:
                    return False
            return True
        else:
            return False

    def __str__(self) -> str:
        return f'{self._values}'

    def __float__(self) -> float:
        if self.length == 1:
            return float(self._values[0])
        raise ValueError('float() argument must be a vector with one element')

    def __list__(self) -> list:
        return self._values.copy()

    def __len__(self) -> int:
        return self.length

    def append(self, value: T) -> None:
        if isinstance(value, (int, float)):
            self._values.append(value)
            self.length += 1
        else:
            raise TypeError(f"value to be added must be int or float not {type(value)}")

    def copy(self) -> "vector[T]":
        return vector(self._values)

    def norm(self) -> float:
        result = 0
        for i in range(self.length):
            result += self._values[i] ** 2
        return sqrt(result)

    def insert(self, index: SupportsIndex, value: T):
        if isinstance(value, (int, float)):
            self._values.insert(index, value)
        else:
            raise TypeError(f"value to be inserted must be an integer or float not {type(value)}")


class matrix(MutableSequence[T], Generic[T]):

    @overload
    def __init__(self, rows: SupportsIndex, cols: SupportsIndex):
        ...

    @overload
    def __init__(self, values: list[list[T]]):
        ...

    def __init__(self, *args: Union[list[list[T]], list[SupportsIndex, SupportsIndex]]):
        if len(args) == 1 and isinstance(args[0], list):
            self._values = copy.deepcopy(args[0])
            self.rows = len(self._values)
            self.cols = len(self._values[0])

        elif len(args) == 2 and all(isinstance(x, SupportsIndex) for x in args):
            self._values = [[0] * args[1] for _ in range(args[0])]
            self.rows = args[0]
            self.cols = args[1]
        else:
            raise ValueError("Invalid values passed")

    def __getitem__(self, pos: Union[tuple[SupportsIndex, SupportsIndex], SupportsIndex]) -> T:
        if isinstance(pos, tuple):
            return self._values[pos[0]][pos[1]]
        elif isinstance(pos, SupportsIndex):
            return self._values[pos]
        else:
            raise TypeError("index must be tuple of int or int")

    def __setitem__(self, index: Union[tuple[SupportsIndex, SupportsIndex], SupportsIndex], value: T):
        if isinstance(value, (int, float)):
            if isinstance(index, tuple) and all(isinstance(i, SupportsIndex) for i in index):
                self._values[index[0]][index[1]] = value
            elif isinstance(index, SupportsIndex):
                self._values[index] = value
            else:
                raise TypeError(f"index must be tuple[SupportIndex, SupportIndex] not {type(index)}")
        else: 
            raise TypeError(f"value must be an integer or float not {type(value)}")

    def __delitem__(self, args: tuple[str, SupportsIndex]):
        if args[0] == "row":
            del self._values[args[1]]
            self.rows -= 1
        elif args[0] == "col":
            for i in range(self.rows):
                del self._values[i][args[1]]
            self.cols -= 1

    def __len__(self) -> int:
        return self.rows

    @overload
    def multiply(self, other: "matrix") -> "matrix":
        ...

    @overload
    def multiply(self, other: vector[T]) -> vector[T]:
        ...

    def multiply(self, other: Union[vector[T], "matrix"]) -> Union[vector[T], "matrix"]:
        if isinstance(other, vector):
            if self.cols != other.length:
                raise IndexError(
                    f"The multiplier vector must have dimension ({self.cols}, ), but has ({other.length, })")
            else:
                new_vector = vector(self.rows)
                for i in range(new_vector.length):
                    for j in range(self.cols):
                        new_vector[i] += self._values[i][j] * other[j]
                return new_vector
        elif isinstance(other, matrix):
            if self.cols != other.rows:
                raise IndexError(
                    f"The multiplier matrix must have dimension ({self.cols}, n), but has ({self.rows, self.cols})")
            else:
                new_matrix = matrix(self.rows, other.cols)
                for i in range(new_matrix.rows):
                    for j in range(new_matrix.cols):
                        for k in range(self.cols):
                            new_matrix[i, j] += self._values[i][k] * \
                                other[k, j]
                return new_matrix
        else:
            raise TypeError(
                f"The multiplier must be vector or matrix not {type(other)}")

    def transpose(self) -> "matrix":
        t_matrix = matrix(self.cols, self.rows)
        for i in range(t_matrix.rows):
            for j in range(t_matrix.cols):
                t_matrix[i, j] = self[j, i]
        return t_matrix

    def inverse(self) -> 'matrix':
        """Inverse matrix by Gauss method"""
        if self.rows != self.cols:
            raise ValueError("Not a square matrix")
        adh_matrix = self.copy()
        add_values = [0] * adh_matrix.rows
        for i in range(adh_matrix.rows):
            add_values[i] = 1
            adh_matrix.insert(1, adh_matrix.cols, add_values)
            add_values[i] = 0

        # forward trace
        for k in range(self.rows):
            # 1) Swap k-row with one of the underlying if m[k, k] = 0
            nonzero_row = pick_nonzero_row(adh_matrix, k)
            
            if adh_matrix.rows == nonzero_row:
                raise ValueError("Singular matrix")

            if nonzero_row != k:
                adh_matrix[k], adh_matrix[nonzero_row] = adh_matrix[nonzero_row], adh_matrix[k]

            # 2) Make diagonal element equals to 1
            if adh_matrix[k, k] != 1:
                divider = adh_matrix[k, k]
                for col in range(k, adh_matrix.cols):
                    adh_matrix[k, col] *= 1 / divider

            # 3) Make all underlying elements in column equal to zero
            for row in range(k+1, adh_matrix.rows):
                multiplier = adh_matrix[row, k]
                for col in range(adh_matrix.cols):
                    adh_matrix[row, col] -= adh_matrix[k, col] * multiplier

        # backward trace
        for k in range(adh_matrix.rows - 1, 0, -1):
            for row in range(k - 1, -1, -1):
                if adh_matrix[row, k]:
                    multiplier = adh_matrix[row, k]
                    for col in range(k, adh_matrix.cols):
                        adh_matrix[row, col] -= adh_matrix[k, col] * multiplier

        for _ in range(self.cols):
            del adh_matrix['col', 0]
        return adh_matrix

    def __str__(self) -> str:
        result_str = "["
        for i in range(self.rows - 1):
            result_str += f"{self._values[i]}\n "
        result_str += f"{self._values[i+1]}]"
        return result_str
    
    def __list__(self) -> list:
        return copy.deepcopy(self._values)

    def copy(self) -> 'matrix':
        return matrix(self._values)

    def insert(self, dimension: SupportsIndex, index: SupportsIndex, values: list[T]):
        if dimension == 0:
            if len(values) == self.cols:
                self._values.insert(index, values)
                self.rows = len(self._values)
            else:
                raise ValueError(
                    "The size of the row to be inserted is greater than the size of the matrix rows")
        elif dimension == 1:
            if len(values) == self.rows:
                for row in range(self.rows):
                    self._values[row].insert(index, values[row])
                self.cols = len(self._values[0])
            else:
                raise ValueError(
                    "The size of column to be inserted is greater than the size of the matrix columns")


def pick_nonzero_row(m: "matrix", k: SupportsIndex):
    j = k
    while j < m.rows and not m[j, k]:
        j += 1
    return j
