"""
Component metadata and sets
---------------------------

- The Meta validation and its serialize and deserialize round trip
- The InputSet, ConstantSet, and DataSet containers of a component
"""

import pytest

from amigo.modeling.component import ConstantSet, DataSet, InputSet, Meta
from amigo.modeling.meta import VarType
from amigo.modeling.expressions import ConstNode, Expr, VarNode

# Meta validates its fields at construction


class TestMetaValidation:
    """Meta construction rules and the serialize and deserialize round trip"""

    def test_name_with_double_underscore_raises(self):
        """Names with a double underscore are reserved for amigo"""
        with pytest.raises(ValueError):
            Meta("bad__name", VarType.INPUT)

    def test_lower_greater_than_upper_raises(self):
        """The bounds must satisfy lower <= upper"""
        with pytest.raises(ValueError):
            Meta("x", VarType.INPUT, lower=10.0, upper=1.0)

    def test_input_value_outside_bounds_raises(self):
        """The initial value of an input must lie within its bounds"""
        with pytest.raises(ValueError):
            Meta("x", VarType.INPUT, value=5.0, lower=0.0, upper=4.0)

    def test_unknown_kwarg_raises(self):
        """An unknown keyword argument is rejected instead of ignored"""
        with pytest.raises(ValueError):
            Meta("x", VarType.INPUT, unknown_kwarg=42)

    def test_serialize_deserialize_roundtrip(self):
        """Deserialize rebuilds every field that serialize wrote"""
        m = Meta("y", VarType.CONSTRAINT, lower=-1.0, upper=1.0, value=0.0)
        data = m.serialize()

        # Deserialize pops from the dict, so it gets a copy
        m2 = Meta.deserialize(dict(data))
        data2 = m2.serialize()

        for name in data:
            if m.is_value_set(name):
                assert data[name] == data2[name]


# InputSet holds the named active variables of a component


class TestInputSet:
    """InputSet add, item access, and the error for a missing key"""

    def test_add_increases_len(self):
        inputs = InputSet()

        assert len(inputs) == 0

        inputs.add("x")

        assert len(inputs) == 1

    def test_getitem_returns_expr_with_active_varnode(self):
        inputs = InputSet()
        inputs.add("x")

        expr = inputs["x"]

        assert isinstance(expr, Expr)
        assert isinstance(expr.node, VarNode)
        assert expr.node.active is True

    def test_add_checks_the_start_against_the_bounds(self):
        """An input added with a start outside its bounds is rejected"""
        inputs = InputSet()

        with pytest.raises(ValueError):
            inputs.add("x", value=5.0, lower=0.0, upper=4.0)

    def test_getitem_missing_key_raises_keyerror(self):
        inputs = InputSet()

        with pytest.raises(KeyError):
            inputs["missing"]


# ConstantSet and DataSet hold the named constants and the passive data


class TestConstantSet:
    """ConstantSet stores scalar constants and rejects other shapes"""

    def test_add_stores_constnode_with_correct_value(self):
        consts = ConstantSet()
        consts.add("c", 3.14)

        expr = consts["c"]

        assert isinstance(expr, Expr)
        assert isinstance(expr.node, ConstNode)
        assert expr.node.value == 3.14

    def test_add_non_scalar_shape_raises_valueerror(self):
        consts = ConstantSet()

        with pytest.raises(ValueError):
            consts.add("c", 1.0, shape=(3,))


class TestDataSet:
    """DataSet stores inactive variables with the declared shape"""

    def test_add_stores_varnode_inactive_with_shape(self):
        data = DataSet()
        data.add("d", shape=(3,))

        expr = data["d"]
        node = expr.node

        assert isinstance(expr, Expr)
        assert isinstance(node, VarNode)
        assert node.active is False
        assert node.shape == (3,)
