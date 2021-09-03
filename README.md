<!-- font: brandon -->

#<p align="center" >Particle-based Smooth Particle Hydrodynamics</p>

&ensp;&ensp; This project, **_Smooth Particle Hydrodynamics_**, is a course work for the course **High-End Simulation in Practice (HESPA)** for Computational Engineering at **Friedrich-Alexander-Universität Erlangen-Nürnberg**.

####1 Introduction

&ensp;&ensp; **Smoothed-particle hydrodynamics (SPH)** is a particle based Lagrangian method. It is a meshfree and highly flexible computational method used for visualising the mechanics of continuous fluid media. It was initially developed for visualisation astronomical events (formation of Galaxies, stars, plants, occurrence of supernova...) by Gingold and Monaghan and Lucy in 1977 [1,2]. Now-a-days it has a wide range of applications including study of volcanos, ocean currents, fluid flow, etc,.

####2 Underlaying Function in SPH

&ensp;&ensp; As name suggests, SPH smoothens the fluid properties a particle over the neighbouring particles within a support region with radius h using a weighted function or smoothening function **$W(r,h)$**.
The integral interpolation of any continuous field, **$f(x)$** is given by,
    $$f(x_{i}) = \int_\Omega f(x_j) W(r,h) dx_j  $$ where, $$ r = x_i - x_j $$

The integral is approximated using a Riemann summation over the particles as:
$$ f_{i} = \sum_{\forall j} m_j \  \frac{f_j}{\rho_j} \  W(r,\ h) $$

The gradient and laplacian of the summation function is given by:
$$ \nabla f_{i} = \sum_{\forall j} \ m_j \ \frac{f_j}{\rho_j} \ \nabla\ W(r,\ h) $$ $$ \nabla^2 f_{i} = \sum_{\forall j} \ m_j \ \frac{f_j}{\rho_j} \ \nabla^2\ W(r,\ h) $$
When using SPH to derive fluid equations for particles, these equations are not guaranteed to satisfy certain physical principals such as symmetry of forces and conservation of momentum.

####3 Modelling Fluids with Particle

&ensp;&ensp;In the Eulerian (grid based) formulation, isothermal fluids
are governed by a velocity field $v$, a density field $\rho$ and a
pressure field $p$. The evolution of these quantities is given by **$Navier-Stokes\ equation\ for\ incompressible\ force$**. The first equation,conservation of mass: 
$$ \frac{\partial\rho}{\partial t} \ + \ \nabla \ . \ (\rho v)\ =\ 0  $$ and second equation, conservation of momentum: 
$$ \rho\ (\frac{\partial v}{\partial t} + v\ \nabla\ .\ v) = -\ \nabla p\ +\ \rho g\ +\ \mu\nabla^2v$$ where **$g$** is an external force field and **$\mu$**, the dynamic viscosity of the fluid. 
&ensp;&ensp;The SPH method simplifies these equation significantly by considering dynamic particle frame of refeerance instead of static grid. Since the particel are finite and no particle escapes the conservation of mass is guaranteed. So the second equation can be rewriten as: $$ \underbrace{\rho\ (\frac{D v}{D t})}_{\rho\ a} = \underbrace{-\ \nabla p\ +\ \rho g\ +\ \mu\nabla^2v}_{f}$$ $$ a_i\ =\ \frac{D v_i}{D t}\ =\ \frac{f_i}{\rho} $$ $$ f_i\ =\ f_i^{pressure}\ +\ f_i^{external}\ +\ f_i^{viscosity}\ +\ f_i^{surface}  $$<br>


#####3.1 Mass Density Field

&ensp;&ensp; The mass associated with a particle remains same throughout the simulation but the density has to be calculated everytime. As discussed, the formula of density,
$$ \rho_i = \sum_{\forall j} m_j\ \frac{\rho_j}{\rho_j}\ W(r,\ h)\ =\ \sum_{\forall j} m_j\ W(r,\ h)  $$ The smoothening kernel which is best suited for the density field calculation was deviced by Müller, Charypar and Gross[3]
$$ W_{poly6}(r,\ h)\ = \frac{315}{64\pi h^9} \begin{cases}(h^2-\|r\|^2)^3 & 0 \leq \|r\| \leq 0 \\ 0 & else\end{cases} $$

#####3.2 Pressure Field

&ensp;&ensp; Applying SPH to second NS equation we get,

$$ f_i^{pressure}\ =\ -\nabla p\ =\ -\sum_{\forall j}\ m_j\ \frac{p_j}{\rho_j}\ \nabla\ W(r,\ h)$$ 

&ensp;&ensp; Unfortunately, this force is not symmetric as can be seen when only two particles interact. Since the gradient of the kernel is zero at its center, particle i only uses the pressure of particle j to compute its pressure force and vice versa. Because the pressures at the locations of the two particles are not equal in general, the pressure forces will not be symmetric. A very simple solution which is best suited for the speed and stability of simulation is by averaging the pressures:

$$ f_i^{pressure}\ =\ -\sum_{\forall j \not ={i}}\ m_j\ \frac{p_i\ +\ p_j}{2\ .\ \rho_j}\ \nabla\ W(r,\ h) $$ The pressure can be computed via the ideal gas state equation,
$$ p = k\rho $$ But the litrature suggests a modification, $$ p_i = k(\rho_i-\rho_0) $$ where **$k$** is ideal gass costant $(=3J)$ and **$\rho_0$** is the rest desnsity of fluid.

The smoothening kernel which is best suited for the pressure field calculation was proposed by Debrun,


$$\nabla\ W_{spiky}(r,\ h)\ = -\frac{45}{\pi h^6 \|r\|} r \begin{cases}(h-\|r\|)^2 & 0 \leq \|r\| \leq 0 \\ 0 & else\end{cases}$$

#####3.3 External Field 

&ensp;&ensp; The externel field can take any kind of force such as gravity, magnetic and electic fields force (for electron soup modelling), centrifugal force, collision or user interaction.

#####3.4 Viscosity Field

Applying the SPH to the viscosity term $\mu\nabla^2v$ again yields asymmetric forces
$$ f_i^{viscosity}=\mu\nabla^2v = \mu\ \sum_{\forall j}\ m_j\ \frac{v_j}{\rho_j}\ \nabla^2\ W(r,\ h) $$ &ensp;&ensp;Since viscosity forces are only dependent on velocity differences and not on absolute velocities, there is a natural way to symmetrize the viscosity forces by using velocity differences:
$$ f_i^{viscosity}=\mu\ \sum_{\forall j \not ={i}}\ m_j\ \frac{v_j-v_i}{\rho_j}\ \nabla^2\ W(r,\ h) $$ 

&ensp;&ensp; Viscosity is a phenomenon that is caused by friction and, thus, decreases the fluid’s kinetic energy by converting it into heat. Therefore, viscosity should only have a smoothing effect on the velocity field. However, if a standard kernel is used for viscosity, the resulting viscosity forces do not always have this property. For two particles that get close to each other, the Laplacian of the smoothed velocity field (on which viscosity forces depend) can get negative resulting in forces that increase their relative velocity. Thus, for the computation of viscosity forces Müller[3] designed a third kernel:

$$\nabla^2\ W_{viscosity}(r,\ h)\ = \frac{45}{\pi h^6} \begin{cases}(h-\|r\|) & 0 \leq \|r\| \leq 0 \\ 0 & else\end{cases}$$

#####3.5 Surface Tension

&ensp;&ensp; Surface tension forces is modelled based on proposal by Morris[4]. Molecules in a fluid are subject to attractive forces from neighboring molecules. Inside the fluid these intermolecular forces are equal in all directions and the net force is unbalanced and acts in the direction of surface normal towards fluid, minimising the surface curvature.
&ensp;&ensp; In literature, the surface is found using a field refered as color which is 1 at particle center and 0 elsewhere.
$$ c_s(r) = \sum_{\forall j}\ \frac{m_j}{\rho_j}\ W(r,\ h) $$ and surface normal ppointing into the fluid,
$$ n\ =\ \nabla c_s $$ the divergence of n measures the curvature of the surface,
$$ k = \frac{-\nabla^2 c_s}{\|n\|} $$ The minus is necessary to get positive curvature for convex fluid volumes. The force acting near the surface,
$$ f_i^{surface} = \sigma k n = -\sigma\ \nabla^2 c_s \frac{n}{\|n\|} $$

#####3.6 Leapfrog Integration Scheme

&ensp;&ensp; **Leapfrog integration** is a method for numerically integrating second order differential equations particularly in the case of a dynamical system of classical mechanics. It is similar to the velocity Verlet method, which is a variant of **Verlet integration**. Leapfrog integration is equivalent to updating positions **$x(t)$** and velocities **$v(t)=\dot x(t)$** at interleaved time points, staggered in such a way that they "leapfrog" over each other.
The "kick-draft-kick" for of Leapfrog Integration Scheme can be writen as :
$$ v_{t+0.5\Delta t} = v_t + a_t \frac{\Delta t}{2} $$ $$ x_{t+\Delta t} = x_t + v_{t+0.5\Delta t} \Delta t $$ $$ v_{t+0.5\Delta t} = v_t + a_{t+\Delta t} \frac{\Delta t}{2} $$

<br><br><br><br>
#### References:
[1] R. A. Gingold and J. J. Monaghan. "Smoothed particle hydrodynamics". Monthly Notices of the Royal Astronomical Society 181, pp.375-389, 1977.

[2] L. B. Lucy. "A numerical approach to the testing of the fussion hypothesis". Astrophysical Journal 82, pp. 1013-1024, 1977.

[3] Müller, Charypar and Gross, "Particle-Based Fluid Simulation for Interactive Applications". The Eurographics Association 2003.

[4] N. Foster and R. Fedkiw. Practical animation of liquids. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 23–30. ACMPress, 2001.

[5] keel, R. D., "Variable Step Size Destabilizes the Stömer/Leapfrog/Verlet Method", BIT Numerical Mathematics, Vol. 33, 1993, p. 172–175.