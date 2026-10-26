from dataclasses import dataclass
from enum import Enum, auto


class Space(Enum):
    H1 = auto()
    L2 = auto()
    HDIV = auto()
    HCURL = auto()
    CONST = auto()


class Conformity(Enum):
    GLOBAL = auto()
    COMPONENT = auto()


class IntegrandFormulation(Enum):
    POTENTIAL = auto()
    WEAK = auto()


@dataclass(frozen=True)
class FunctionSpace:
    func_space: Space
    degree: int = 1
    conformity: Conformity = Conformity.GLOBAL


class SolutionSpace:
    def __init__(self, mapping, degree=1):
        self._space_mapping = {
            "H1": Space.H1,
            "L2": Space.L2,
            "H(div)": Space.HDIV,
            "H(curl)": Space.HCURL,
            "const": Space.CONST,
        }

        self.fields = []  # (name, FunctionSpace) in declaration order
        self.names = {}  # FunctionSpace -> list(solution names)
        self.comp_names = {}  # FunctionSpace -> comp_name

        for comp_name in mapping:
            if len(mapping[comp_name]) == 0:
                raise ValueError(f"Empty specification for {comp_name}")
            elif len(mapping[comp_name]) > 1:
                raise ValueError(f"Overspecification for {comp_name}")

            [(names, space)] = mapping[comp_name].items()
            if isinstance(names, str):
                names = (names,)
            if isinstance(space, Space):
                space = FunctionSpace(space)
            elif isinstance(space, str):
                try:
                    # Enum lookup is by name, not value: auto() values are ints.
                    space = FunctionSpace(self._space_mapping[space], degree=degree)
                except KeyError:
                    raise ValueError(f"Unknown function space '{space}'")
            elif not isinstance(space, FunctionSpace):
                raise TypeError(f"Expected FunctionSpace or str, got {type(space)}")

            self.comp_names[space] = comp_name
            self.names.setdefault(space, []).extend(names)
            self.fields.extend((name, space) for name in names)

    def get_spaces(self):
        return list(self.names)

    def get_names(self, space: FunctionSpace):
        return self.names.get(space, [])

    def get_component_name(self, input: FunctionSpace | str):
        """Get the component name from the function space or variable name"""
        if isinstance(input, str):
            return self.comp_names.get(self.get_space(input), None)
        else:
            return self.comp_names.get(input, None)

    def get_space(self, name: str):
        for field_name, space in self.fields:
            if field_name == name:
                return space
        return None
