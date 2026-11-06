import amigo as am
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()


all_inputs = {
    "x": {},
    "z": {"shape": 2, "value": [1, 1]},
    "y": {"shape": 2, "value": [1, 1]},
}

disp1 = am.ExpressionComponent(
    name="Disp1",
    inputs=all_inputs,
    constraints={"c1": "z[0] ** 2 + z[1] + x - 0.2 * y[1] - y[0]"},
)

disp2 = am.ExpressionComponent(
    name="Disp2",
    inputs={"z": {"shape": 2, "value": [1, 1]}, "y": {"shape": 2, "value": [1, 1]}},
    constraints={"c2": "sqrt(y[0]) + z[0] + z[1] - y[1]"},
)

obj = am.ExpressionComponent(
    name="Objective",
    inputs=all_inputs,
    objective={"obj": "x**2 + z[1] + y[0] + exp(-y[1])"},
)

con1 = am.ExpressionComponent(
    name="Con1", inputs={"y": {"shape": 2}}, constraints={"g1": "3.16 - y[0]"}
)

con2 = am.ExpressionComponent(
    name="Con2", inputs={"y": {"shape": 2}}, constraints={"g1": "y[1] - 24.0"}
)

model = am.Model("sellar")
model.add_component("disp1", 1, disp1)
model.add_component("disp2", 1, disp2)
model.add_component("obj", 1, obj)
model.add_component("con1", 1, con1)
model.add_component("con2", 1, con2)

model.link_by_name()
if args.build:
    model.build_module()

model.initialize()

x = model.create_vector()
opt = am.Optimizer(model, x)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_line_search_iterations": 10,
        "max_iterations": 500,
    }
)
