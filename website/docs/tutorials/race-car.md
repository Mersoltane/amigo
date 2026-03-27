---
sidebar_position: 5
---

# Race Car Minimum Lap Time

## 1. Introduction

Lap time simulation of racing vehicles is nowadays widely used to predict car performance on race tracks and to optimize the vehicle setup before track tests. The two most common methods used to perform such simulations are the quasi-steady-state and the optimal control approaches. In the quasi-steady-state method, a fixed trajectory is provided as input data, the path is divided into small segments, and the vehicle maximum speed is calculated at each corner apex. Starting from these known points, acceleration and braking zones are reconstructed by forward and backward integration. Such simulations are fast to compute and very robust, which is why they remain the most widely used performance tools in racing teams. However, the fixed trajectory represents a fundamental limitation for the accuracy of the result, since the method cannot discover that a different line through a corner sequence might yield a faster overall lap.

The optimal control approach removes this limitation entirely. Rather than prescribing a trajectory, the racing line emerges as part of the solution. The optimizer determines simultaneously all driver inputs (steering, throttle, braking) and the resulting vehicle path, subject to the full nonlinear vehicle dynamics and all physical constraints. The trade-offs that a human driver must resolve by intuition and experience, such as how late to brake, how much to sacrifice corner entry speed for a better exit, or when to use the full width of the track, are handled automatically by the optimization. In practice, the continuous optimal control problem is transcribed into a large but finite nonlinear program (NLP) using direct collocation, and then solved numerically by an interior-point method.

In this work, the minimum lap time problem (MLTP) is solved using a formulation inspired by the work of Christ et al. [1], originally developed at the Technical University of Munich for the Roborace autonomous racing competition. The vehicle is modeled as a planar rigid body with a double-track configuration in which all four wheels are treated independently. Lateral tire forces are computed with an extended Pacejka model that accounts for load-dependent friction degression, and wheel normal loads follow from a quasi-steady-state load transfer model that includes aerodynamic downforce. The track is described in curvilinear coordinates with arc length as the independent variable, so that time itself becomes the quantity to be minimized. Physical limits are enforced through Kamm's friction circle at each wheel, engine power constraints, actuator rate bounds, and track boundary conditions. Because the track is a closed circuit, the solution must be periodic: the vehicle state at the end of the lap must match the state at the start.

The remainder of this document presents the track model, the vehicle model, the optimal control problem formulation, and the numerical method.

## 2. Track Model

### 2.1 Curvilinear Coordinates

The track is described using a **curvilinear coordinate system** that uses the arc length $s$ of a reference line (the track centerline) as the independent variable. This approach, widely used in the racing literature [1, 2, 4], offers several advantages: it provides a compact description of the vehicle's progress along the circuit, simplifies the treatment of track boundary constraints, and naturally handles the periodicity of a closed lap.

<div style={{textAlign: 'center'}}>
<img src={require('./TrackModel.png').default} alt="Track model" style={{width: '480px'}} />

*Figure 2.1: Curvilinear coordinate system on the Berlin Formula E circuit. The vehicle position is described by the arc length $s$ along the centerline and the lateral displacement $n$. The relative heading angle $\xi$ is defined between the vehicle longitudinal axis and the centerline tangent, $\beta$ is the body sideslip angle, and $R = 1/\kappa$ is the local radius of curvature. The track boundaries $N_l + N_r$ define the admissible lateral range.*
</div>

The vehicle position relative to the track is described by the **lateral displacement** $n$ from the reference line (positive to the left) and the **relative heading angle** $\xi$ between the vehicle's longitudinal axis and the tangent to the reference line. At any point $s$, the track geometry is characterized by the **curvature** $\kappa(s)$, equal to the inverse of the local radius of curvature $R(s)$, and the **track half-widths** $w_\text{left}(s)$ and $w_\text{right}(s)$ to the left and right of the centerline.

The time evolution of the curvilinear coordinates is given by [1]:

$$
\dot{s} = \frac{v \cos(\xi + \beta)}{1 - n\kappa}, \qquad \dot{n} = v \sin(\xi + \beta), \qquad \dot{\xi} = \omega_z - \kappa \frac{v \cos(\xi + \beta)}{1 - n\kappa}
$$

where $v$ is the vehicle speed, $\beta$ is the body sideslip angle, and $\omega_z$ is the yaw rate.

### 2.2 Change of Independent Variable

In race line optimization, it is standard practice to use the **path coordinate** $s$ as the independent variable rather than time $t$ [1, 2, 4]. The time to be minimized therefore becomes a dependent variable. This choice eliminates $s$ as a state variable, reducing the problem size. It also enables a simple description of track boundaries and lap periodicity, and allows position-dependent parameters such as curvature and friction to be introduced directly.

The transformation requires the **slowness factor**:

$$
S_F = \frac{dt}{ds} = \frac{1 - n\kappa}{v \cos(\xi + \beta)}
$$

which relates time increments to arc-length increments. The total lap time is then:

$$
T_\text{lap} = \int_0^{s_f} S_F \, ds
$$

All state equations formulated in the time domain $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$ are converted to the arc-length domain via $\mathbf{x}' = S_F \cdot \mathbf{f}(\mathbf{x}, \mathbf{u})$, where primes denote differentiation with respect to $s$.

### 2.3 Track Data

The recording of the racetrack centerline and road boundaries can be performed using differential GPS or 2D-LiDAR. The raw data consists of centerline coordinates $(x_i, y_i)$ with corresponding left and right half-widths. From these, the curvature is computed numerically:

$$
\kappa = \frac{x' y'' - y' x''}{(x'^2 + y'^2)^{3/2}}
$$

The curvature of the raw centerline typically exhibits high-frequency oscillations from measurement noise, which is counterproductive for the robustness and numerical efficiency of the optimization. The reference line is therefore preprocessed by **Gaussian smoothing** before being passed to the optimization problem. The track is then resampled to uniformly spaced arc-length nodes.

## 3. Vehicle Model

### 3.1 Point Mass Model

The complexity of a vehicle model for lap time optimization is determined by the number of dynamic degrees of freedom retained in the formulation. The simplest useful representation is the point mass model, a planar rigid body with three degrees of freedom: longitudinal velocity, lateral velocity (expressed through the body sideslip angle), and yaw rate. Despite its name, this model is not a literal point mass, since it accounts for the moment of inertia in yaw and treats all four wheels independently through a double-track configuration. The name reflects the fact that out-of-plane motions (roll, pitch, heave) and wheel rotational dynamics are neglected, so that the chassis behaves as a rigid body constrained to the road plane.

<div style={{textAlign: 'center'}}>
<img src={require('./VehiclePointMassModel.png').default} alt="Point mass model" style={{width: '320px'}} />

*Figure 3.1: Top-down view of the double-track point mass model. Each wheel produces a longitudinal force $F_x$ and a lateral force $F_y$ independently, with $l_f$ and $l_r$ denoting the distances from the center of gravity to the front and rear axles.*
</div>

This level of fidelity is well suited for the minimum lap time problem. It captures the essential coupling between longitudinal and lateral tire forces, the asymmetric load transfer during combined cornering and acceleration, and the yaw dynamics that govern transient behavior through corner entries and exits. More complex models with additional degrees of freedom (such as the seven or eight DOF formulations that add roll, pitch, and individual wheel spin) provide higher physical accuracy but introduce significantly more variables and constraints, which increases the computational burden without necessarily improving the optimized racing line by a meaningful margin.

### 3.2 Double-Track Model

The formulation adopted here follows Christ et al. [1], who developed this model for a Formula E vehicle in the context of the Roborace autonomous racing competition. The double-track layout treats all four wheels independently, which is essential for capturing the asymmetric tire loading that occurs during combined cornering and acceleration. Each wheel generates its own longitudinal force $F_x$ and lateral force $F_y$, determined by the local slip angle, normal load, and friction conditions. The steering angle $\delta_w$ acts on the front wheels, while the rear wheels remain fixed in their orientation relative to the chassis. The free-body diagram with all modeling quantities is shown in Figure 3.2.

<div style={{textAlign: 'center'}}>
<img src={require('./DOFmodel.png').default} alt="Double-track model" style={{width: '520px'}} />

*Figure 3.2: Free-body diagram of the double-track vehicle model showing tire forces ($F_{x,ij}$, $F_{y,ij}$), slip angles ($\alpha_{ij}$), wheel velocities ($v_{ij}$), steering angle ($\delta_w$), sideslip angle ($\beta$), yaw rate ($\dot\psi$), and the geometric parameters: front and rear track widths ($T_F$, $T_R$) and axle distances ($l_f$, $l_r$).*
</div>

### 3.3 State and Control Variables

The dynamics are described by a set of first-order ordinary differential equations. The **state variables** are:

$$
\mathbf{x} = [v, \; \beta, \; \omega_z, \; n, \; \xi]^T
$$

| Symbol | Description | Unit |
|--------|-------------|------|
| $v$ | Vehicle speed (at the center of gravity) | m/s |
| $\beta$ | Body sideslip angle | rad |
| $\omega_z$ | Yaw rate | rad/s |
| $n$ | Lateral displacement from the reference line | m |
| $\xi$ | Relative heading angle | rad |

The **control variables** are:

$$
\mathbf{u} = [\delta, \; F_\text{drive}, \; F_\text{brake}, \; \Gamma_y]^T
$$

| Symbol | Description | Unit |
|--------|-------------|------|
| $\delta$ | Front steering angle | rad |
| $F_\text{drive}$ | Total driving force ($\geq 0$) | N |
| $F_\text{brake}$ | Total braking force ($\leq 0$) | N |
| $\Gamma_y$ | Lateral load transfer force | N |

The splitting of the longitudinal force into separate driving and braking variables avoids the need for non-smooth functions to distinguish between acceleration and deceleration [1].

### 3.4 Equations of Motion

The three dynamic equations describe the longitudinal, lateral, and yaw motion with respect to the center of gravity:

$$
\dot{v} = \frac{1}{m} \Big[ (F_{x,rl} + F_{x,rr})\cos\beta + (F_{x,fl} + F_{x,fr})\cos(\delta - \beta) + (F_{y,rl} + F_{y,rr})\sin\beta - (F_{y,fl} + F_{y,fr})\sin(\delta - \beta) - F_\text{drag}\cos\beta \Big]
$$

$$
\dot{\beta} = -\omega_z + \frac{1}{mv} \Big[ -(F_{x,rl} + F_{x,rr})\sin\beta + (F_{x,fl} + F_{x,fr})\sin(\delta - \beta) + (F_{y,rl} + F_{y,rr})\cos\beta + (F_{y,fl} + F_{y,fr})\cos(\delta - \beta) + F_\text{drag}\sin\beta \Big]
$$

$$
\dot{\omega}_z = \frac{1}{J_{zz}} \Big[ (F_{x,rr} - F_{x,rl})\frac{t_{w,r}}{2} - (F_{y,rl} + F_{y,rr}) l_r + \Big((F_{x,fr} - F_{x,fl})\cos\delta + (F_{y,fl} - F_{y,fr})\sin\delta\Big)\frac{t_{w,f}}{2} + \Big((F_{y,fl} + F_{y,fr})\cos\delta + (F_{x,fl} + F_{x,fr})\sin\delta\Big) l_f \Big]
$$

Here $m$ is the vehicle mass, $J_{zz}$ the yaw moment of inertia, $l_f$ and $l_r$ the distances from the center of gravity to the front and rear axles, and $t_{w,f}$, $t_{w,r}$ the front and rear track widths. The aerodynamic drag force is $F_\text{drag} = \tfrac{1}{2} c_d \rho A v^2$.

Combined with the curvilinear kinematics from Section 2.1, the lateral displacement and heading evolve as:

$$
n' = v \sin(\xi + \beta) \cdot S_F, \qquad \xi' = \omega_z \cdot S_F - \kappa
$$

### 3.5 Longitudinal Tire Forces

The braking and driving forces are distributed to the individual wheels via static distribution coefficients $k_\text{drive}$ and $k_\text{brake}$:

$$
F_{x,fj} = \tfrac{1}{2} k_\text{drive} F_\text{drive} + \tfrac{1}{2} k_\text{brake} F_\text{brake} - \tfrac{1}{2} f_r m g \frac{l_r}{L}
$$

$$
F_{x,rj} = \tfrac{1}{2} (1 - k_\text{drive}) F_\text{drive} + \tfrac{1}{2} (1 - k_\text{brake}) F_\text{brake} - \tfrac{1}{2} f_r m g \frac{l_f}{L}
$$

where $j \in \{l, r\}$ denotes left or right, $f_r$ is the rolling resistance coefficient, and $L = l_f + l_r$ is the wheelbase.

### 3.6 Lateral Tire Forces: Extended Pacejka Model

The lateral tire forces are determined using an **extended version of Pacejka's Magic Formula** that includes a degressive behavior against the wheel load:

$$
F_{y,ij} = \mu_{ij} F_{z,ij} \left(1 + \varepsilon_i \frac{F_{z,ij}}{F_{z0,i}}\right) \sin\!\Big(C_i \arctan\!\big(B_i \alpha_{ij} - E_i (B_i \alpha_{ij} - \arctan(B_i \alpha_{ij}))\big)\Big)
$$

The coefficient $\varepsilon \leq 0$ introduces a **degressive tire behavior** against the normal load: as the wheel load increases, the friction force grows less than proportionally. This is a critical physical effect, because without it the simple Pacejka model significantly overestimates the lateral force capacity at high normal loads.

The **tire slip angles** $\alpha_{ij}$ are computed from kinematic relations:

$$
\alpha_{fl/fr} = \delta - \arctan\!\left(\frac{l_f \omega_z + v \sin\beta}{v \cos\beta \mp \tfrac{1}{2} t_{w,f} \omega_z}\right)
$$

$$
\alpha_{rl/rr} = \arctan\!\left(\frac{l_r \omega_z - v \sin\beta}{v \cos\beta \mp \tfrac{1}{2} t_{w,r} \omega_z}\right)
$$

where the upper sign corresponds to the left wheel and the lower sign to the right wheel.

### 3.7 Wheel Normal Loads

The normal loads are computed under a **quasi-steady-state** assumption that neglects suspension dynamics. They depend on the static weight distribution, longitudinal load transfer from acceleration, aerodynamic downforce, and lateral load transfer:

$$
F_{z,fl} = \frac{mgl_r}{2L} - \frac{h_\text{cog}}{2L} m a_x - k_\text{roll} \Gamma_y + \tfrac{1}{2} c_{L,f} q_\text{dyn}
$$

$$
F_{z,fr} = \frac{mgl_r}{2L} - \frac{h_\text{cog}}{2L} m a_x + k_\text{roll} \Gamma_y + \tfrac{1}{2} c_{L,f} q_\text{dyn}
$$

$$
F_{z,rl} = \frac{mgl_f}{2L} + \frac{h_\text{cog}}{2L} m a_x - (1 - k_\text{roll}) \Gamma_y + \tfrac{1}{2} c_{L,r} q_\text{dyn}
$$

$$
F_{z,rr} = \frac{mgl_f}{2L} + \frac{h_\text{cog}}{2L} m a_x + (1 - k_\text{roll}) \Gamma_y + \tfrac{1}{2} c_{L,r} q_\text{dyn}
$$

where $q_\text{dyn} = \tfrac{1}{2}\rho A v^2$ is the dynamic pressure times frontal area, $h_\text{cog}$ is the center of gravity height, $k_\text{roll}$ is the roll moment distribution (fraction at front axle), and $c_{L,f}$, $c_{L,r}$ are the front and rear aerodynamic downforce coefficients.

The approximate longitudinal acceleration entering the load transfer is:

$$
m a_x \approx F_\text{drive} + F_\text{brake} - F_\text{drag} - f_r m g
$$

The lateral load transfer variable $\Gamma_y$ must satisfy the equilibrium:

$$
\Gamma_y = \frac{h_\text{cog}}{\bar{t}_w} \Big(F_{y,rl} + F_{y,rr} + (F_{x,fl} + F_{x,fr})\sin\delta + (F_{y,fl} + F_{y,fr})\cos\delta\Big)
$$

where $\bar{t}_w = \tfrac{1}{2}(t_{w,f} + t_{w,r})$ is the mean track width. This is enforced as an **equality constraint** in the optimization.

## 4. Optimal Control Problem

### 4.1 Objective

The primary objective is to minimize the total lap time:

$$
\min_{\mathbf{x}(\cdot), \, \mathbf{u}(\cdot)} \quad J = \int_0^{s_f} S_F(s) \, ds
$$

A **control regularization** term is added to penalize abrupt changes in steering and longitudinal force, promoting smooth and physically realizable inputs:

$$
J_\text{reg} = \sum_{k=0}^{N-1} \Big[ r_\delta \big(\Delta\delta_k\big)^2 + r_F \big(\Delta F_k\big)^2 \Big]
$$

where $r_\delta$ and $r_F$ are penalty weights. The total cost is $J + J_\text{reg}$.

### 4.2 Inequality Constraints

The optimization is subject to a number of physical inequality constraints at each mesh point.

**Kamm's friction circle** ensures that the combined longitudinal and lateral force at each wheel does not exceed the friction limit:

$$
F_{x,ij}^2 + F_{y,ij}^2 \leq (\mu_{ij} F_{z,ij})^2, \qquad ij \in \{fl, \, fr, \, rl, \, rr\}
$$

**Engine power limit**:

$$
v \cdot F_\text{drive} \leq P_\text{max}
$$

**Actuator rate constraints** limit how fast the controls can change between consecutive intervals:

$$
\left|\frac{\Delta\delta}{\Delta t}\right| \leq \frac{\delta_\text{max}}{T_\delta}, \qquad \frac{\Delta F_\text{drive}}{\Delta t} \leq \frac{F_\text{drive,max}}{T_\text{drive}}, \qquad \frac{-\Delta F_\text{brake}}{\Delta t} \leq \frac{|F_\text{brake,min}|}{T_\text{brake}}
$$

**Track boundary constraints** prevent the vehicle from leaving the road:

$$
-w_\text{right}(s) + \frac{b_\text{veh}}{2} \leq n(s) \leq w_\text{left}(s) - \frac{b_\text{veh}}{2}
$$

### 4.3 Boundary Conditions

Since the track is a closed loop, the solution must satisfy **periodic boundary conditions**:

$$
\mathbf{x}(0) = \mathbf{x}(s_f)
$$

The vehicle must arrive at the start/finish line with the same velocity, sideslip angle, yaw rate, lateral position, and heading as when it departed.

### 4.4 Variable Bounds

The state and control variables are subject to box constraints reflecting physical limits:

| Variable | Lower bound | Upper bound |
|---|---|---|
| $v$ | 1 m/s | $v_\text{max}$ |
| $\beta$ | $-\pi/2$ | $\pi/2$ |
| $\omega_z$ | $-2$ rad/s | $2$ rad/s |
| $n$ | $-(w_r - b_\text{veh}/2)$ | $w_l - b_\text{veh}/2$ |
| $\xi$ | $-\pi/2$ | $\pi/2$ |
| $\delta$ | $-\delta_\text{max}$ | $\delta_\text{max}$ |
| $F_\text{drive}$ | 0 | $F_\text{drive,max}$ |
| $F_\text{brake}$ | $F_\text{brake,min}$ | 0 |

## 5. Vehicle Parameters

The vehicle parameters correspond to a Formula E class racing car (Table A1 of [1]):

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Vehicle mass | $m$ | 1200 | kg |
| CG to front axle | $l_f$ | 1.5 | m |
| CG to rear axle | $l_r$ | 1.4 | m |
| Track width, front | $t_{w,f}$ | 1.6 | m |
| Track width, rear | $t_{w,r}$ | 1.5 | m |
| Vehicle width | $b_\text{veh}$ | 2.0 | m |
| CG height | $h_\text{cog}$ | 0.4 | m |
| Yaw moment of inertia | $J_{zz}$ | 1260 | kg m$^2$ |
| Frontal area | $A$ | 1.0 | m$^2$ |
| Drag coefficient | $c_d$ | 1.4 | -- |
| Downforce coeff., front | $c_{L,f}$ | 2.4 | -- |
| Downforce coeff., rear | $c_{L,r}$ | 3.0 | -- |
| Rolling resistance coeff. | $f_r$ | 0.010 | -- |
| Air density | $\rho$ | 1.2041 | kg/m$^3$ |
| Max engine power | $P_\text{max}$ | 270 | kW |
| Max driving force | $F_\text{drive,max}$ | 7100 | N |
| Max braking force | $F_\text{brake,min}$ | $-$20000 | N |
| Max steering angle | $\delta_\text{max}$ | 0.4 | rad |
| Max velocity | $v_\text{max}$ | 42.5 | m/s |
| Drive force distribution | $k_\text{drive}$ | 0.0 (RWD) | -- |
| Brake force distribution | $k_\text{brake}$ | 0.7 | -- |
| Roll moment distribution | $k_\text{roll}$ | 0.5 | -- |

### Pacejka Tire Parameters (Table A2 of [1])

| Parameter | Front | Rear |
|---|---|---|
| Stiffness factor $B$ | 9.62 | 8.62 |
| Shape factor $C$ | 2.59 | 2.65 |
| Curvature factor $E$ | 1.0 | 1.0 |
| Nominal load $F_{z0}$ (N) | 3000 | 3000 |
| Load degression $\varepsilon$ | $-$0.0813 | $-$0.1263 |
| Friction coefficient $\mu$ | 1.0 | 1.0 |

## 6. Numerical Method

The continuous optimal control problem is converted to a finite-dimensional NLP using **direct collocation**. The track is divided into $N$ intervals of equal arc length $\Delta s = s_f / N$, with a target step size of $\Delta s \approx 3$ m. The states are defined at the $N$ mesh nodes (with cyclic wrap-around), and controls are **piecewise constant** over each interval.

The dynamics are enforced through **trapezoidal collocation** defect constraints:

$$
\mathbf{x}_{k+1} - \mathbf{x}_k - \frac{\Delta s}{2}\Big(\mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, \kappa_k) + \mathbf{f}(\mathbf{x}_{k+1}, \mathbf{u}_k, \kappa_{k+1})\Big) = 0, \qquad k = 0, \ldots, N-1
$$

The lap time is approximated by trapezoidal quadrature:

$$
T_\text{lap} \approx \sum_{k=0}^{N-1} \frac{\Delta s}{2} \big(S_{F,k} + S_{F,k+1}\big)
$$

The resulting NLP is solved by an **interior-point method** with a filter-based line search and second-order corrections.

## References

<span id="ref-1">1.</span> Christ, F., Wischnewski, A., Heilmeier, A., & Lohmann, B. (2021). Time-optimal trajectory planning for a race car considering variable tyre-road friction coefficients. *Vehicle System Dynamics*, 59(4), 588--612.

<span id="ref-2">2.</span> Casanova, D. (2000). *On Minimum Time Vehicle Manoeuvring: The Theoretical Optimal Lap*. PhD thesis, Cranfield University.

<span id="ref-3">3.</span> Kelly, D. P. (2008). *Lap Time Simulation with Transient Vehicle and Tyre Dynamics*. PhD thesis, Cranfield University.

<span id="ref-4">4.</span> Perantoni, G., & Limebeer, D. J. N. (2014). Optimal control for a Formula One car with variable parameters. *Vehicle System Dynamics*, 52(5), 653--678.

<span id="ref-5">5.</span> Dal Bianco, N., Lot, R., & Gadola, M. (2018). Minimum time optimal control simulation of a GP2 race car. *Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering*, 232(9), 1180--1195.

<span id="ref-6">6.</span> Pacejka, H. B. (2012). *Tire and Vehicle Dynamics* (3rd ed.). Butterworth-Heinemann.

<span id="ref-7">7.</span> Wachter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. *Mathematical Programming*, 106(1), 25--57.
