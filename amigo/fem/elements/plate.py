from ..element import MITCTyingStrain, MITCElement, MITCStrainComponent
from ..cell_types import CellType
from ..fem_space import Space, FunctionSpace, SolutionSpace, Conformity
from ..basis import make_basis, H1Value
from ..quadrature import make_quadrature


class MITC4PlateTying(MITCTyingStrain):
    def __init__(self):
        # Two tying points each for g23 and g13, respectively
        self.tying_points = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_tying_points(self):
        return self.tying_points

    def eval_tying_strain(self, idx, geo, soln):
        rx = soln["rx"].value
        ry = soln["ry"].value

        # Get the derivatives in the computational coordinates
        w_1, w_2 = soln["w"].grad

        # Derivatives wrt computational coordinates
        x_1, x_2 = geo["x"].grad
        y_1, y_2 = geo["y"].grad

        if idx < 2:
            g23 = w_2 + rx * x_2 + ry * y_2
            return g23
        elif idx < 4:
            g13 = w_1 + rx * x_1 + ry * y_1
            return g13

        raise ValueError("Tying point index out of range")

    def interp_and_transform(self, pt, detJ, J, Jinv, e):
        # Interpolate the tensorial components of the tying strains
        g23 = 0.5 * ((1.0 - pt[0]) * e[0] + (1.0 + pt[0]) * e[1])
        g13 = 0.5 * ((1.0 - pt[1]) * e[2] + (1.0 + pt[1]) * e[3])

        gxz = Jinv[0][0] * g13 + Jinv[1][0] * g23
        gyz = Jinv[0][1] * g13 + Jinv[1][1] * g23

        # Convert from tensorial to real strain
        out = {}
        out["gxz"] = MITCStrainComponent(value=gxz)
        out["gyz"] = MITCStrainComponent(value=gyz)

        return out


class MITCPlate:
    def __init__(
        self,
        constitutive=None,
        E: float = 70e9,
        nu: float = 0.3,
        ks: float = 5.0 / 6.0,
        t: float = 0.01,
    ):
        self.E = E
        self.nu = nu
        self.ks = ks
        self.t = t

        self.constitutive = constitutive
        if self.constitutive is None:
            self.constitutive = self.constitutive_default

        return

    def constitutive_default(self, soln, data=None, geo=None):

        G = 0.5 * self.E / (1.0 + self.nu)
        D = self.E / (1.0 - self.nu**2)

        D11 = D * self.t**3 / 12.0
        D12 = self.nu * D11
        D22 = D11
        D66 = 0.5 * D11 * (1 - self.nu)

        As44 = self.ks * G * self.t
        As55 = As44

        return (D11, D12, D22, D66), (As44, As55)

    @staticmethod
    def get_default_spaces():
        # Set the function spaces
        H1 = FunctionSpace(func_space=Space.H1, degree=1, conformity=Conformity.GLOBAL)

        # 6 DOF/node: u, v, w translations + rx, ry, rz global rotations
        soln_space = SolutionSpace({"soln": {("w", "rx", "ry"): H1}})

        # Separate spaces for the continuous x, y, z coordinates and the normals
        # that are discontinuous across components
        geo_space = SolutionSpace({"geo": {("x", "y"): H1}})

        # By default, no data space object
        data_space = SolutionSpace({})

        # Compute the local normals/weights
        output_space = SolutionSpace({})

        return soln_space, geo_space, data_space, output_space

    def create_element(
        self,
        name: str = "Plate",
        ctype: CellType = CellType.QUADRILATERAL,
        soln_space: SolutionSpace | None = None,
        geo_space: SolutionSpace | None = None,
        data_space: SolutionSpace | None = None,
        integrand=None,
    ):

        soln, geo, data, _ = MITCPlate.get_default_spaces()
        if soln_space is None:
            soln_space = soln
        if data_space is None:
            data_space = data
        if geo_space is None:
            geo_space = geo

        # Get the basis and quadrature objects needed for the element
        ctype = CellType.QUADRILATERAL
        soln_basis = make_basis(soln_space, ctype, kind="input")
        geo_basis = make_basis(geo_space, ctype, kind="data")
        data_basis = None
        quadrature = make_quadrature(soln_space, ctype)
        mitc = MITC4PlateTying()

        if integrand is None:
            integrand = self.potential

        # Create the MITC4 element
        plate_obj = MITCElement(
            name, soln_basis, data_basis, geo_basis, quadrature, mitc, integrand
        )

        return plate_obj

    def potential(self, soln, data=None, geo=None):
        """Strain energy density (integrand of TPE equation)"""

        rx_x, rx_y = soln["rx"].grad
        ry_x, ry_y = soln["ry"].grad

        gxz = soln["gxz"].value
        gyz = soln["gyz"].value

        # components of kappa vector
        kx = rx_x
        ky = ry_y
        kxy = rx_y + ry_x

        (D11, D12, D22, D66), (As44, As55) = self.constitutive(soln, data=data, geo=geo)

        # bending strain energy
        Ub = 0.5 * (D11 * kx**2 + D22 * ky**2 + 2 * D12 * kx * ky + D66 * kxy**2)

        # shear strain energy
        Us = 0.5 * (As44 * gyz**2 + As55 * gxz**2)

        return Ub + Us
