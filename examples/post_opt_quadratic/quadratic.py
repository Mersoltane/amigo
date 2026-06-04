import amigo as am


class Quadratic(am.Component):
    def __init__(self, a=1.0, b=3.0):
        super().__init__()

        self.add_input("x1", value=0.0, lower=-10, upper=10)
        self.add_input("x2", value=0.0, lower=-10, upper=10)
        self.add_data("a", value=a)
        self.add_data("b", value=b)
        self.add_objective("obj")
        self.add_output("f")

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]

        a = self.data["a"]
        b = self.data["b"]

        self.objective["obj"] = x1**2 + x1 * x2 + x2**2 - a * x1 - b * x2

    def compute_output(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]

        self.outputs["f"] = x1**2 + x2**2


a = 1.0
b = 3.0
model = am.Model("quadratic")
model.add_component("quad", 1, Quadratic(a=a, b=b))

model.build_module()
model.initialize()

# Create the design variable vector and provide an initial guess
x = model.create_vector()

opt = am.Optimizer(model, x)
opt_data = opt.optimize()

dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
    of="quad.f", wrt=["quad.a", "quad.b"]
)

# f^{star} = (5 * a**2 - 8 * a * b + 5 * b**2) / 9.0
print(dfdx[of_map["quad.f"], wrt_map["quad.a"]] - (10 * a - 8 * b) / 9)
print(dfdx[of_map["quad.f"], wrt_map["quad.b"]] - (10 * b - 8 * a) / 9)
