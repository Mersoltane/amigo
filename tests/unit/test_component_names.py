"""A component refuses a name that another of its quantities already carries."""

import pytest

import amigo as am


class Comm(am.Component):
    def __init__(self):
        super().__init__()
        self.add_data("gain", shape=(15, 15))
        self.add_input("Dr")


def test_an_input_cannot_take_the_name_of_a_data_table():
    with pytest.raises(ValueError, match="already data"):
        Comm().add_input("gain")


def test_a_constraint_cannot_take_the_name_of_an_input():
    with pytest.raises(ValueError, match="already an input"):
        Comm().add_constraint("Dr")


def test_distinct_names_are_accepted():
    comp = Comm()
    comp.add_input("Gt")
    comp.add_constraint("res", shape=2)
    assert "Gt" in comp.inputs.meta and "res" in comp.constraints.meta
