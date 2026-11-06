"""
Model helpers
-------------

- The GlobalIndexPool and the parsers of variable strings and attribute paths
- The record of the entries set through a ModelVector
- The compiled extension is stubbed by conftest.py when it is missing, so these run without a C++ build
"""

import ast

import numpy as np
import pytest

import amigo as am
from amigo.modeling.model import GlobalIndexPool, ModelVector, _parse_attribute_path, _parse_var_expr


# GlobalIndexPool allocates contiguous index arrays


class TestGlobalIndexPool:
    """GlobalIndexPool shapes, values, non-overlapping ranges, and counter"""

    def test_first_allocate_shape_and_values(self):
        """The first allocation of shape (3,) holds 0, 1, and 2"""
        pool = GlobalIndexPool()

        result = pool.allocate((3,))

        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_successive_calls_non_overlapping(self):
        """Successive allocations return ranges that do not overlap"""
        pool = GlobalIndexPool()

        first = pool.allocate((3,))
        second = pool.allocate((2,))

        np.testing.assert_array_equal(first, [0, 1, 2])
        np.testing.assert_array_equal(second, [3, 4])

    def test_allocate_2d_shape(self):
        """An allocation of shape (2, 3) has that shape"""
        pool = GlobalIndexPool()

        result = pool.allocate((2, 3))

        assert result.shape == (2, 3)

    def test_counter_advances_after_each_allocation(self):
        """The counter moves past every allocation"""
        pool = GlobalIndexPool()

        assert pool.counter == 0

        pool.allocate((3,))

        assert pool.counter == 3

        # Three plus six entries
        pool.allocate((2, 3))

        assert pool.counter == 9


# The parsers turn variable strings and attribute paths into path and index pairs


class TestParseVarExpr:
    """_parse_var_expr on dotted paths, slices, and invalid input"""

    def test_simple_attribute_path(self):
        """A dotted path has no indices"""
        parsed = _parse_var_expr("model.group.var")
        path = parsed[0]
        indices = parsed[1]

        assert path == ["model", "group", "var"]
        assert indices is None

    def test_full_slice(self):
        """A full slice gives one slice of everything"""
        parsed = _parse_var_expr("model.group.var[:]")
        path = parsed[0]
        indices = parsed[1]

        assert path == ["model", "group", "var"]
        assert indices == (slice(None, None, None),)

    def test_multi_index_slice_and_int(self):
        """A slice and an integer give both indices in order"""
        parsed = _parse_var_expr("model.group.var[:, 0]")
        path = parsed[0]
        indices = parsed[1]

        assert path == ["model", "group", "var"]
        assert indices == (slice(None, None, None), 0)

    def test_invalid_expression_raises_value_error(self):
        """A string that is not an attribute or subscript expression is rejected"""
        with pytest.raises(ValueError):
            _parse_var_expr("not a valid expression!")


class TestParseAttributePath:
    """_parse_attribute_path rebuilds the name segments of an AST node in order"""

    def test_two_part_path(self):
        """The path a.b gives its two names"""
        node = ast.parse("a.b", mode="eval").body

        result = _parse_attribute_path(node)

        assert result == ["a", "b"]


class TestModelVectorAssigned:
    """ModelVector records the entries set through it so optimizers apply only those"""

    def make_vector(self, size=5, blank=True):
        """A model vector over an amigo vector, blank or wrapping existing values"""
        return ModelVector(None, am.Vector(size), blank=blank)

    def test_blank_vector_has_no_assignment(self):
        assert not self.make_vector().get_assigned().any()

    def test_wrapped_vector_counts_as_fully_set(self):
        vec = self.make_vector(blank=False)
        vec[1] = 2.0

        assert vec.get_assigned() is None

    def test_index_and_slice_assignments_are_marked(self):
        vec = self.make_vector()
        vec[1] = 2.0
        vec[np.array([3, 4])] = [5.0, 6.0]

        assert vec.get_assigned().tolist() == [False, True, False, True, True]

    def test_full_slice_marks_every_entry(self):
        vec = self.make_vector()
        vec[:] = 0.0

        assert vec.get_assigned().all()

    def test_copy_marks_every_entry(self):
        vec = self.make_vector()
        vec.copy(self.make_vector())

        assert vec.get_assigned().all()
