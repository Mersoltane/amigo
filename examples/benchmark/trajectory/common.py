"""
Benchmark transcription
-----------------------

- Hermite Simpson collocation with the states, rates, and controls at the nodes and the midpoints
- The literature family starts from the guess published with each problem
- The linear family moves every state from its initial value to its final value
- A state with a free final value holds its initial value in the linear family
- The constant family holds every state at its initial value
- The tf_half and tf_double families fly the linear family over half or twice the final time
- The rates of every start match the dynamics at the guess
"""

import amigo as am
import numpy as np


# Hermite Simpson defects of one state over one interval of a free final time
class HermiteSimpson(am.Component):
    def __init__(self, num_intervals):
        super().__init__()

        self.add_data("inv_steps", value=1.0 / num_intervals)

        self.add_input("tf")

        self.add_input("q1")
        self.add_input("q2")
        self.add_input("qm")

        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_input("qmdot")

        self.add_constraint("herm")
        self.add_constraint("simp")

        return

    def compute(self):
        inv_steps = self.data["inv_steps"]

        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        qm = self.inputs["qm"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]
        qmdot = self.inputs["qmdot"]

        dt = tf * inv_steps

        # Midpoint interpolation and Simpson quadrature over one interval
        self.constraints["herm"] = qm - 0.5 * (q1 + q2) - (dt / 8.0) * (q1dot - q2dot)
        self.constraints["simp"] = q2 - q1 - (dt / 6.0) * (q1dot + 4.0 * qmdot + q2dot)

        return


# Hermite Simpson defects of one state over one interval of a fixed length
class HermiteSimpsonFixed(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_data("dt", value=dt)

        self.add_input("q1")
        self.add_input("q2")
        self.add_input("qm")

        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_input("qmdot")

        self.add_constraint("herm")
        self.add_constraint("simp")

        return

    def compute(self):
        dt = self.data["dt"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        qm = self.inputs["qm"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]
        qmdot = self.inputs["qmdot"]

        # Midpoint interpolation and Simpson quadrature over one interval
        self.constraints["herm"] = qm - 0.5 * (q1 + q2) - (dt / 8.0) * (q1dot - q2dot)
        self.constraints["simp"] = q2 - q1 - (dt / 6.0) * (q1dot + 4.0 * qmdot + q2dot)

        return


def link_collocation(model, node, mid, hs, num_states, num_intervals):
    """Link the states and rates at the nodes and the midpoints to the collocation"""
    for i in range(num_states):
        start = i * num_intervals
        end = (i + 1) * num_intervals

        # Link the states at the nodes and the midpoints
        model.link(f"{node}.x[:{num_intervals}, {i}]", f"{hs}.q1[{start}:{end}]")
        model.link(f"{node}.x[1:, {i}]", f"{hs}.q2[{start}:{end}]")
        model.link(f"{mid}.x[:, {i}]", f"{hs}.qm[{start}:{end}]")

        # Link the state rates at the nodes and the midpoints
        model.link(f"{node}.xdot[:-1, {i}]", f"{hs}.q1dot[{start}:{end}]")
        model.link(f"{node}.xdot[1:, {i}]", f"{hs}.q2dot[{start}:{end}]")
        model.link(f"{mid}.xdot[:, {i}]", f"{hs}.qmdot[{start}:{end}]")

    return


def family_states(guess, x0, xf, tau):
    """States of the linear or the constant family at the normalized times"""
    start = np.asarray(x0, dtype=float)

    # A state with a free final value holds its initial value
    target = np.array(start)

    for i in range(len(xf)):
        if xf[i] is not None:
            target[i] = xf[i]

    # The constant family holds every state at its initial value
    if guess == "constant":
        target = start

    return start[None, :] + np.outer(tau, target - start)


def family_final_time(guess, tf):
    """Nominal final time scaled by the final time families"""
    scale = 1.0

    if guess == "tf_half":
        scale = 0.5
    elif guess == "tf_double":
        scale = 2.0

    return scale * tf
