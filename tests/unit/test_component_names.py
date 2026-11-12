"""
Component names
---------------

- A component refuses a name that another of its quantities already carries
- A component refuses a name that the generated C++ code cannot declare
"""

import pytest

import amigo as am


class Comm(am.Component):
    """A component with one data table and one input"""

    def __init__(self):
        super().__init__()

        self.add_data("gain", shape=(15, 15))
        self.add_input("Dr")

        return


def test_an_input_cannot_take_the_name_of_a_data_table():
    with pytest.raises(ValueError, match="already data"):
        Comm().add_input("gain")


def test_a_constraint_cannot_take_the_name_of_an_input():
    with pytest.raises(ValueError, match="already an input"):
        Comm().add_constraint("Dr")


def test_a_constant_cannot_take_a_cpp_keyword():
    with pytest.raises(ValueError, match="generated C"):
        Comm().add_constant("switch", value=1.0)


def test_an_input_cannot_take_a_name_that_is_no_identifier():
    with pytest.raises(ValueError, match="generated C"):
        Comm().add_input("two words")


def test_distinct_names_are_accepted():
    comp = Comm()

    comp.add_input("Gt")
    comp.add_constraint("res", shape=2)

    assert "Gt" in comp.inputs.meta and "res" in comp.constraints.meta
