import amigo as am
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..expressions import Expr


class FiniteElement(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        integrand,
        test_basis=None,
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.integrand = integrand
        self.test_basis = test_basis

        # From BasisCollection
        self.soln_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)
        if self.data_basis is not None:
            self.data_basis.add_declarations(self)

        # Add constraint declarations
        if self.test_basis is not None:
            self.test_basis.add_declarations(self)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        return

    def compute(self, **args):

        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        if self.test_basis is not None:
            test_xi = self.test_basis.eval(self, quad_point)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = None
        if self.data_basis is not None:
            data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        Jdict = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(Jdict, soln_xi)
        data_phys = None
        if self.data_basis is not None:
            data_phys = self.data_basis.transform(Jdict, data_xi)

        if self.test_basis is not None:
            # Transform the test space derivatives into the physical space
            test_phys = self.test_basis.transform(Jdict, test_xi)

            # Don't add the alpha coefficient - this will all be proportaional to the
            # multipliers through the test function
            detJ = Jdict["detJ"]
            weight = quad_weight * detJ
            return weight * self.integrand(
                test_phys, soln_phys, data=data_phys, geo=geo
            )
        else:
            # Include the alpha coefficient since we're adding terms to the objective
            detJ = Jdict["detJ"]
            weight = self.alpha * quad_weight * detJ
            return weight * self.integrand(soln_phys, data=data_phys, geo=geo)


class FiniteElementOutput(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        output_names,
        output_function,
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.output_names = output_names
        self.output_function = output_function

        # From BasisCollection
        self.soln_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)
        if self.data_basis is not None:
            self.data_basis.add_declarations(self)

        # add the output declarations, assuming scalar functions
        for name in self.output_names:
            self.add_output(name)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        return

    def compute_output(self, **args):
        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = None
        if self.data_basis is not None:
            data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        Jdict = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(Jdict, soln_xi)
        data_phys = None
        if self.data_basis is not None:
            data_phys = self.data_basis.transform(Jdict, data_xi)

        # Add the contributions directly to the Lagrangian
        outputs = self.output_function(soln_phys, data=data_phys, geo=geo)

        for name in self.output_names:
            if name in outputs:
                detJ = Jdict["detJ"]
                self.outputs[name] = quad_weight * detJ * outputs[name]
            else:
                self.outputs[name] = 0.0
        return


class FiniteElementFieldOutput(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        output_basis,
        output_function,
    ):
        super().__init__(name=name)

        self.soln_basis = soln_basis
        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.output_basis = output_basis
        self.output_function = output_function

        # From BasisCollection
        self.soln_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)
        if self.data_basis is not None:
            self.data_basis.add_declarations(self)

        # add the output declarations, assuming scalar functions
        self.output_basis.add_declarations(self)

        # Set the arguments to the compute function for each quadrature point
        args = []
        for n in range(len(self.output_basis.basis[0].layout.pts)):
            args.append({"n": n})
        self.set_args(args)

        return

    def compute_output(self, n=None):
        point = self.output_basis.basis[0].layout.pts[n]

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, point)
        data_xi = None
        if self.data_basis is not None:
            data_xi = self.data_basis.eval(self, point)
        geo = self.geo_basis.eval(self, point)

        # Perform the mapping from computational to physical coordinates (u)
        Jdict = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(Jdict, soln_xi)
        data_phys = None
        if self.data_basis is not None:
            data_phys = self.data_basis.transform(Jdict, data_xi)

        # Add the contributions directly to the Lagrangian
        outputs = self.output_function(soln_phys, data=data_phys, geo=geo)

        for name in outputs:
            self.outputs[f"{name}{n}"] = outputs[name]

        return


@dataclass
class MITCStrainComponent:
    value: Expr | None = None


class MITCTyingStrain(ABC):
    @abstractmethod
    def get_tying_points(self):
        """Return all the tying points within the element"""
        pass

    @abstractmethod
    def eval_tying_strain(self, index, geo, soln):
        """Evaluate the covariant component of the strain at the tying point"""
        return 0.0

    @abstractmethod
    def interp_and_transform(self, pt, Jinv, tying_strains):
        """Interpolate and transform the tying strain"""
        return {}


class MITCElement(FiniteElement):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        mitc,
        integrand,
        test_basis=None,
    ):
        if not isinstance(mitc, MITCTyingStrain):
            raise ValueError("MITCElement requires instance of MITCTyingStrain")

        super().__init__(
            name,
            soln_basis,
            data_basis,
            geo_basis,
            quadrature,
            integrand,
            test_basis=test_basis,
        )

        self.mitc = mitc

    def compute(self, **args):
        # Compute the tensorial strain components at tying points
        tensorial_strains = []
        for index, pt in enumerate(self.mitc.get_tying_points()):
            soln_xi = self.soln_basis.eval(self, pt)
            geo = self.geo_basis.eval(self, pt)

            # Append the tying strain
            tensorial_strains.append(self.mitc.eval_tying_strain(index, geo, soln_xi))

        # Compute the test tensorial strain at tying points - this only works for linear strain expressions
        test_tensorial_strains = []
        if self.test_basis is not None:
            for index, pt in enumerate(self.mitc.get_tying_points()):
                test_xi = self.test_basis.eval(self, pt)
                geo = self.geo_basis.eval(self, pt)

                # Append the tying strain
                test_tensorial_strains.append(
                    self.mitc.eval_tying_strain(index, geo, test_xi)
                )

        # Now get the quadrature point
        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        if self.test_basis is not None:
            test_xi = self.test_basis.eval(self, quad_point)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = None
        if self.data_basis is not None:
            data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        Jdict = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(Jdict, soln_xi)
        data_phys = None
        if self.data_basis is not None:
            data_phys = self.data_basis.transform(Jdict, data_xi)

        # Add the interpolated tying strains
        soln_phys.update(
            self.mitc.interp_and_transform(quad_point, Jdict, tensorial_strains)
        )

        if self.test_basis is not None:
            # Transform the test space derivatives into the physical space
            test_phys = self.test_basis.transform(Jdict, test_xi)

            # Add the interpolated tying strains
            test_phys.update(
                self.mitc.interp_and_transform(
                    quad_point, Jdict, test_tensorial_strains
                )
            )

            # Don't add the alpha coefficient - this will all be proportaional to the
            # multipliers through the test function
            detJ = Jdict["detJ"]
            weight = quad_weight * detJ
            return weight * self.integrand(
                test_phys, soln_phys, data=data_phys, geo=geo
            )
        else:
            # Add the contributions directly to the Lagrangian
            detJ = Jdict["detJ"]
            weight = self.alpha * quad_weight * detJ
            return weight * self.integrand(soln_phys, data=data_phys, geo=geo)


class MITCElementOutput(FiniteElementOutput):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        mitc,
        output_names,
        output_function,
    ):
        if not isinstance(mitc, MITCTyingStrain):
            raise ValueError("MITCElement requires instance of MITCTyingStrain")

        super().__init__(
            name,
            soln_basis,
            data_basis,
            geo_basis,
            quadrature,
            output_names,
            output_function,
        )
        self.mitc = mitc

        return

    def compute_output(self, **args):
        # Compute the tensorial strain components at tying points
        tensorial_strains = []
        for index, pt in enumerate(self.mitc.get_tying_points()):
            soln_xi = self.soln_basis.eval(self, pt)
            geo = self.geo_basis.eval(self, pt)

            # Append the tying strain
            tensorial_strains.append(self.mitc.eval_tying_strain(index, geo, soln_xi))

        # Now get the quadrature point
        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = None
        if self.data_basis is not None:
            data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        Jdict = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(Jdict, soln_xi)
        data_phys = None
        if self.data_basis is not None:
            data_phys = self.data_basis.transform(Jdict, data_xi)

        # Add to the physics
        soln_phys.update(
            self.mitc.interp_and_transform(quad_point, Jdict, tensorial_strains)
        )

        # Add the contributions directly to the Lagrangian
        outputs = self.output_function(soln_phys, data=data_phys, geo=geo)

        for name in self.output_names:
            if name in outputs:
                detJ = Jdict["detJ"]
                self.outputs[name] = quad_weight * detJ * outputs[name]
            else:
                self.outputs[name] = 0.0
        return
