import ast
from collections.abc import Mapping, Sequence

from .component import Component

from .unary_operations import (
    abs,
    fabs,
    sqrt,
    sin,
    asin,
    cos,
    acos,
    tan,
    atan,
    sinh,
    asinh,
    cosh,
    acosh,
    tanh,
    atanh,
    exp,
    log,
    atan2,
    min2,
    max2,
    passive,
)

_AMIGO_FUNCTIONS = {
    "abs": abs,
    "fabs": fabs,
    "sqrt": sqrt,
    "sin": sin,
    "asin": asin,
    "cos": cos,
    "acos": acos,
    "tan": tan,
    "atan": atan,
    "sinh": sinh,
    "asinh": asinh,
    "cosh": cosh,
    "acosh": acosh,
    "tanh": tanh,
    "atanh": atanh,
    "exp": exp,
    "log": log,
    "atan2": atan2,
    "min2": min2,
    "max2": max2,
    "passive": passive,
}


class _ExpressionEvaluator(ast.NodeVisitor):
    _binary_operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }

    _unary_operators = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def __init__(self, namespace, functions=None):
        self.namespace = namespace
        self.functions = dict(functions or {})

    def evaluate(self, expression):
        if not isinstance(expression, str):
            return expression

        tree = ast.parse(expression, mode="eval")
        return self.visit(tree.body)

    def visit_Name(self, node):
        if node.id not in self.namespace:
            raise NameError(f"Unknown variable {node.id!r} in expression")

        return self.namespace[node.id]

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)):
            raise TypeError(f"Unsupported constant {node.value!r} in expression")

        return node.value

    def visit_BinOp(self, node):
        operator = self._binary_operators.get(type(node.op))

        if operator is None:
            raise TypeError(f"Unsupported binary operator " f"{type(node.op).__name__}")

        return operator(
            self.visit(node.left),
            self.visit(node.right),
        )

    def visit_UnaryOp(self, node):
        operator = self._unary_operators.get(type(node.op))

        if operator is None:
            raise TypeError(f"Unsupported unary operator " f"{type(node.op).__name__}")

        return operator(self.visit(node.operand))

    def visit_Call(self, node):
        # Permit sin(x), but reject expressions such as:
        #
        #     obj.method(x)
        #     module.sin(x)
        #
        if not isinstance(node.func, ast.Name):
            raise TypeError(
                "Only direct calls to registered expression functions " "are permitted"
            )

        function_name = node.func.id

        if function_name not in self.functions:
            raise NameError(f"Unknown expression function {function_name!r}")

        if node.keywords:
            raise TypeError(
                f"Expression function {function_name!r} does not "
                "accept keyword arguments"
            )

        arguments = [self.visit(argument) for argument in node.args]
        function = self.functions[function_name]

        try:
            return function(*arguments)
        except TypeError as error:
            raise TypeError(
                f"Invalid arguments for expression function " f"{function_name!r}"
            ) from error

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self._evaluate_index(node.slice)
        return value[index]

    def _evaluate_index(self, node):
        if isinstance(node, ast.Tuple):
            return tuple(self._evaluate_index(item) for item in node.elts)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return node.value

        raise TypeError("Expression indices must be integer constants")

    def generic_visit(self, node):
        raise TypeError(f"Unsupported syntax {type(node).__name__} in expression")


class ExpressionComponent(Component):
    """
    A Component defined using named algebraic expressions.

    Parameters
    ----------
    name : str or None
        Component name.
    inputs : mapping
        Input names mapped to metadata dictionaries. Scalar values are
        interpreted as initial values.
    constants : mapping
        Constant names mapped to metadata dictionaries. Scalar values are
        interpreted as constant values.
    data : mapping
        Data names mapped to metadata dictionaries.
    objective : mapping
        Objective name mapped to a specification containing ``expr`` and
        optional metadata.
    constraints : mapping
        Constraint names mapped to specifications containing ``expr`` and
        optional metadata.
    outputs : mapping
        Output names mapped to specifications containing ``expr`` and
        optional metadata.
    """

    def __init__(
        self,
        name=None,
        inputs=None,
        constants=None,
        data=None,
        objective=None,
        constraints=None,
        outputs=None,
    ):
        super().__init__(name=name)

        inputs = self._normalize_named_specs("inputs", inputs, allow_name_sequence=True)
        constants = self._normalize_named_specs(
            "constants", constants, allow_name_sequence=False
        )
        data = self._normalize_named_specs("data", data, allow_name_sequence=True)
        objective = self._normalize_named_specs(
            "objective", objective, allow_name_sequence=False
        )
        constraints = self._normalize_named_specs(
            "constraints", constraints, allow_name_sequence=False
        )
        outputs = self._normalize_named_specs(
            "outputs", outputs, allow_name_sequence=False
        )

        self._objective_expressions = {}
        self._constraint_expressions = {}
        self._output_expressions = {}

        # Declare independent variables.
        for variable_name, specification in inputs.items():
            metadata = self._normalize_metadata(
                specification,
                scalar_key="value",
            )
            self.add_input(variable_name, **metadata)

        # Declare constants.
        for variable_name, specification in constants.items():
            metadata = self._normalize_metadata(
                specification,
                scalar_key="value",
            )

            if "value" not in metadata:
                raise ValueError(f"Constant {variable_name!r} requires a value")

            self.add_constant(variable_name, **metadata)

        # Declare externally supplied, inactive data.
        for variable_name, specification in data.items():
            metadata = self._normalize_metadata(
                specification,
                scalar_key="value",
            )
            self.add_data(variable_name, **metadata)

        # Declare the objective. ObjectiveSet enforces the one-objective
        # limitation already present in Component.
        for variable_name, specification in objective.items():
            expression, metadata = self._split_expression(
                "objective",
                variable_name,
                specification,
            )
            self.add_objective(variable_name, **metadata)
            self._objective_expressions[variable_name] = expression

        # Declare constraints.
        for variable_name, specification in constraints.items():
            expression, metadata = self._split_expression(
                "constraint",
                variable_name,
                specification,
            )
            self.add_constraint(variable_name, **metadata)
            self._constraint_expressions[variable_name] = expression

        # Declare post-processing outputs.
        for variable_name, specification in outputs.items():
            expression, metadata = self._split_expression(
                "output",
                variable_name,
                specification,
            )
            self.add_output(variable_name, **metadata)
            self._output_expressions[variable_name] = expression

    @staticmethod
    def _normalize_named_specs(
        category,
        specification,
        allow_name_sequence=False,
    ):
        if specification is None:
            return {}

        if isinstance(specification, Mapping):
            return dict(specification)

        if allow_name_sequence:
            if isinstance(specification, (list, tuple)):
                result = {}

                for name in specification:
                    if not isinstance(name, str):
                        raise TypeError(
                            f"Entries in {category} must be strings; "
                            f"received {name!r}"
                        )

                    if name in result:
                        raise ValueError(f"Duplicate name {name!r} in {category}")

                    result[name] = {}

                return result

        expected = "a mapping"
        if allow_name_sequence:
            expected += " or a sequence of names"

        raise TypeError(f"{category} must be {expected}")

    @staticmethod
    def _normalize_metadata(specification, scalar_key):
        if isinstance(specification, Mapping):
            return dict(specification)

        # Convenient shorthand:
        #
        #     inputs={"x": 0.0}
        #     constants={"a": 3.0}
        #
        if isinstance(specification, (int, float)):
            return {scalar_key: specification}

        raise TypeError(
            "Variable specification must be a metadata mapping " "or a scalar value"
        )

    @staticmethod
    def _split_expression(category, name, specification):
        # Convenient shorthand for an expression without metadata:
        #
        #     objective={"f": "x**2"}
        #
        if isinstance(specification, str):
            return specification, {}

        # This can also accept an Amigo Expr directly.
        if not isinstance(specification, Mapping):
            return specification, {}

        metadata = dict(specification)

        if "expr" not in metadata:
            raise ValueError(
                f"{category.capitalize()} {name!r} requires an " "'expr' entry"
            )

        expression = metadata.pop("expr")
        return expression, metadata

    def _make_namespace(self):
        namespace = {}

        for name in self.inputs:
            namespace[name] = self.inputs[name]

        for name in self.constants:
            namespace[name] = self.constants[name]

        for name in self.data:
            namespace[name] = self.data[name]

        return namespace

    def _make_evaluator(self):
        return _ExpressionEvaluator(
            namespace=self._make_namespace(),
            functions=_AMIGO_FUNCTIONS,
        )

    def compute(self, **kwargs):
        evaluator = self._make_evaluator()

        for name, expression in self._objective_expressions.items():
            self.objective[name] = evaluator.evaluate(expression)

        for name, expression in self._constraint_expressions.items():
            self.constraints[name] = evaluator.evaluate(expression)

    def compute_output(self, **kwargs):
        evaluator = self._make_evaluator()

        for name, expression in self._output_expressions.items():
            self.outputs[name] = evaluator.evaluate(expression)
