<div>
<p align="center" >
<h1>Particle-based Smooth Particle Hydrodynamics</h1>
</p>

<p align="center" > 
This project, <b>Smooth Particle Hydrodynamics</b>, is a course work for the course <b>High-End Simulation in Practice (HESPA)</b> for Computational Engineering at <b>Friedrich-Alexander-Universität Erlangen-Nürnberg</b>.
</p>
<hr color=black>
</div>

<div>
<h2> 1 Introduction</h2>
<p align="justify">
<b>Smoothed-particle hydrodynamics (SPH)</b> is a particle based Lagrangian method. It is a meshfree and highly flexible computational method used for visualising the mechanics of continuous fluid media. It was initially developed for visualisation astronomical events (formation of Galaxies, stars, plants, occurrence of supernova...) by Gingold and Monaghan and Lucy in 1977 [1,2]. Now-a-days it has a wide range of applications including study of volcanos, ocean currents, fluid flow, etc.
</p>
</div>

<div>
<h2>2 Underlaying Function in SPH</h2>

<p align="justify">
As name suggests, SPH smoothens the fluid properties a particle over the neighbouring particles within a support region with radius h using a weighted function or smoothening function 
<img src="https://render.githubusercontent.com/render/math?math=$W(r,h)$" alt="$W(r,h)$">.
The integral interpolation of any continuous field, <img src="https://render.githubusercontent.com/render/math?math=$f(x)$" alt="$f(x)$"> is given by,
</p>
<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=$\large f(x_{i}) = \int_\Omega f(x_j) W(r,h) dx_j$" alt="$$f(x_{i}) = \int_\Omega f(x_j) W(r,h) dx_j  $$"> 
</div>
where,
<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=$\large r = x_i - x_j$" alt="$$r = x_i - x_j $$"> 
</div> 
<p align="justify">
The integral is approximated using a <b>Riemann summation over the particles</b> as: 
</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large f_{i} = \sum_{\forall j} m_j \  \frac{f_j}{\rho_j} \  W(r,\ h)$" alt="$$f_{i} = \sum_{\forall j} m_j \  \frac{f_j}{\rho_j} \  W(r,\ h)$$"> </div> 
<p align="justify">
The <b>gradient</b> and <b>laplacian</b> of the summation function is given by:
</p> 
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \nabla f_{i} = \sum_{\forall j} \ m_j \ \frac{f_j}{\rho_j} \ \nabla\ W(r,\ h)$" alt="$$\nabla f_{i} = \sum_{\forall j} \ m_j \ \frac{f_j}{\rho_j} \ \nabla\ W(r,\ h)$$"> </div>
 
<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=$\large \nabla^2 f_{i} = \sum_{\forall j} \ m_j \ \frac{f_j}{\rho_j} \ \nabla^2\ W(r,\ h)$" alt="$$\nabla^2 f_{i} = \sum_{\forall j} \ m_j \ \frac{f_j}{\rho_j} \ \nabla^2\ W(r,\ h)$$"> </div>
 
<p align="justify">When using SPH to derive fluid equations for particles, these equations are <b>not guaranteed</b> to satisfy certain physical principals such as symmetry of forces and conservation of momentum.</p></div>

<div> <h2> 3 Modelling Fluids with Particle</h2>

<p align="justify"> In the Eulerian (grid based) formulation, isothermal fluids are governed by a velocity field <img src="https://render.githubusercontent.com/render/math?math=$v$" alt="$v$">, a density field <img src="https://render.githubusercontent.com/render/math?math=$\rho$" alt="$\rho$"> and a pressure field <img src="https://render.githubusercontent.com/render/math?math=$p$" alt="$p$">. The evolution of these quantities is given by <img src="https://render.githubusercontent.com/render/math?math=$Navier-Stokes\ equation\ for\ incompressible\ force$" alt="$Navier-Stokes\ equation\ for\ incompressible\ force$">. The first equation,conservation of mass:
</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \frac{\partial\rho}{\partial t} \ %2b \ \nabla \ . \ (\rho v)\ =\ 0$" alt="$$\frac{\partial\rho}{\partial t} \ + \ \nabla \ . \ (\rho v)\ =\ 0$$"> </div>

<p align="justify">and second equation, conservation of momentum: </p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \rho\ (\frac{\partial\v}{\partial\t}\%2b\v\nabla.v)=-\nabla\p%2b\ \rho\g\%2b\mu\ \nabla^2v$" alt="$$\rho\ (\frac{\partial v}{\partial t} + v\ \nabla\ .\ v) = -\ \nabla p\ +\ \rho g\ +\ \mu\nabla^2v$$"> </div>

<p align="justify">
where <img src="https://render.githubusercontent.com/render/math?math=$g$" alt="$g$"> is an external force field and <img src="https://render.githubusercontent.com/render/math?math=$\mu$" alt="$\mu$">, the dynamic viscosity of the fluid.
</p> 

<p align="justify">The SPH method simplifies these equation significantly by considering dynamic particle frame of refeerance instead of static grid. Since the particel are finite and no particle escapes the conservation of mass is guaranteed. So the second equation can be rewriten as:</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \rho\ (\frac{D v}{D t}) = -\ \nabla p\ %2b\ \rho g\ %2b\ \mu\nabla^2v$" alt="$$\underbrace{\rho\ (\frac{D v}{D t})}_{\rho\ a} = \underbrace{-\ \nabla p\ +\ \rho g\ +\ \mu\nabla^2v}_{f}$$"> </div>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large a_i\ =\ \frac{D v_i}{D t}\ =\ \frac{f_i}{\rho}$" alt="$$a_i\ =\ \frac{D v_i}{D t}\ =\ \frac{f_i}{\rho}$$"> </div>

<div align="center">
<img src="https://render.githubusercontent.com/render/math?math=$\large f_i\ =\ f_i^{pressure}\ %2b\ f_i^{external}\ %2b\ f_i^{viscosity}\ %2b\ f_i^{surface}$" alt="$$f_i\ =\ f_i^{pressure}\ +\ f_i^{external}\ +\ f_i^{viscosity}\ +\ f_i^{surface}$$"> </div>


<div> <h3> 3.1 Mass Density Field </h3>

<p align="justify">The mass associated with a particle remains same throughout the simulation but the density has to be calculated everytime. As discussed, the formula of density, </p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \rho_i = \sum_{\forall j} m_j\ \frac{\rho_j}{\rho_j}\ W(r,\ h)\ =\ \sum_{\forall j} m_j\ W(r,\ h)$" alt="$$\rho_i = \sum_{\forall j} m_j\ \frac{\rho_j}{\rho_j}\ W(r,\ h)\ =\ \sum_{\forall j} m_j\ W(r,\ h)$$"> </div>

<p align="justify">The smoothening kernel which is best suited for the density field calculation was deviced by Müller, Charypar and Gross[3] </p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large W_{poly6}(r,\ h)\ = \frac{315}{64\pi h^9} \begin{cases}(h^2-\|r\|^2)^3 %26 0 \leq \|r\| \leq 0 \\ 0 %26 else \end{cases}$" alt="$$W_{poly6}(r,\ h)\ = \frac{315}{64\pi h^9} \begin{cases}(h^2-\|r\|^2)^3 & 0 \leq \|r\| \leq 0 \\ 0 & else\end{cases}$$"> </div>
</div>

<div><h3> 3.2 Pressure Field</h3>

<p align="justify">Applying SPH to second NS equation we get, </p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large f_i^{pressure}\ =\ -\nabla p\ =\ -\sum_{\forall j}\ m_j\ \frac{p_j}{\rho_j}\ \nabla\ W(r,\ h)$" alt="$$f_i^{pressure}\ =\ -\nabla p\ =\ -\sum_{\forall j}\ m_j\ \frac{p_j}{\rho_j}\ \nabla\ W(r,\ h)$$"> </div>

<p align="justify">Unfortunately, this force is not symmetric as can be seen when only two particles interact. Since the gradient of the kernel is zero at its center, particle i only uses the pressure of particle j to compute its pressure force and vice versa. Because the pressures at the locations of the two particles are not equal in general, the pressure forces will not be symmetric. A very simple solution which is best suited for the speed and stability of simulation is by averaging the pressures:</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large f_i^{pressure}\ =\ -\sum_{\forall j \not= {i}}\ m_j\ \frac{p_i\ +\ p_j}{2\ .\ \rho_j}\ \nabla\ W(r,\ h)$" alt="$$f_i^{pressure}\ =\ -\sum_{\forall j \not ={i}}\ m_j\ \frac{p_i\ +\ p_j}{2\ .\ \rho_j}\ \nabla\ W(r,\ h)$$"> </div>

<p align="justify">The pressure can be computed via the ideal gas state equation,</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large p = k\rho$" alt="$$p = k\rho$$"> </div>

<p align="justify">But the litrature suggests a modification,</p> 
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large p_i = k(\rho_i-\rho_0)$" alt="$$p_i = k(\rho_i-\rho_0)$$"> </div>

<p align="justify">where <img src="https://render.githubusercontent.com/render/math?math=$k$" alt="$k$"> is ideal gass costant <img src="https://render.githubusercontent.com/render/math?math=$(=3J)$" alt="$(=3J)$"> and <img src="https://render.githubusercontent.com/render/math?math=$\rho_0$" alt="$\rho_0$"> is the rest desnsity of fluid.</p>

<p align="justify">The smoothening kernel which is best suited for the pressure field calculation was proposed by Debrun,</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \nabla\ W_{spiky}(r,\ h)\ = -\frac{45}{\pi h^6 \|r\|} r \begin{cases}(h-\|r\|)^2 %26 0 \leq \|r\| \leq 0 \\ 0 %26 else\end{cases}$" alt="$$\nabla\ W_{spiky}(r,\ h)\ = -\frac{45}{\pi h^6 \|r\|} r \begin{cases}(h-\|r\|)^2 & 0 \leq \|r\| \leq 0 \\ 0 & else\end{cases}$$"> </div>
</div>

<div><h3> 3.3 External Field</h3> 
<p align="justify">The externel field can take any kind of force such as gravity, magnetic and electic fields force (for electron soup modelling), centrifugal force, collision or user interaction.</p>
</div>

<div><h3>3.4 Viscosity Field</h3>

<p align="justify">Applying the SPH to the viscosity term <img src="https://render.githubusercontent.com/render/math?math=$\mu\nabla^2v$" alt="$\mu\nabla^2v$"> again yields asymmetric forces</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large f_i^{viscosity}=\mu\nabla^2v = \mu\ \sum_{\forall j}\ m_j\ \frac{v_j}{\rho_j}\ \nabla^2\ W(r,\ h)$" alt="$$f_i^{viscosity}=\mu\nabla^2v = \mu\ \sum_{\forall j}\ m_j\ \frac{v_j}{\rho_j}\ \nabla^2\ W(r,\ h)$$"> </div>

<p align="justify">Since viscosity forces are only dependent on velocity differences and not on absolute velocities, there is a natural way to symmetrize the viscosity forces by using velocity differences: </p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large f_i^{viscosity}=\mu\ \sum_{\forall j \not ={i}}\ m_j\ \frac{v_j-v_i}{\rho_j}\ \nabla^2\ W(r,\ h)$" alt="$$f_i^{viscosity}=\mu\ \sum_{\forall j \not ={i}}\ m_j\ \frac{v_j-v_i}{\rho_j}\ \nabla^2\ W(r,\ h)$$"> </div>

<p align="justify">Viscosity is a phenomenon that is caused by friction and, thus, decreases the fluid’s kinetic energy by converting it into heat. Therefore, viscosity should only have a smoothing effect on the velocity field. However, if a standard kernel is used for viscosity, the resulting viscosity forces do not always have this property. For two particles that get close to each other, the Laplacian of the smoothed velocity field (on which viscosity forces depend) can get negative resulting in forces that increase their relative velocity. Thus, for the computation of viscosity forces Müller[3] designed a third kernel: </p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large \nabla^2\ W_{viscosity}(r,\ h)\ = \frac{45}{\pi h^6} \begin{cases}(h-\|r\|) %26 0 \leq \|r\| \leq 0 \\ 0 %26 else\end{cases}$" alt="$$\nabla^2\ W_{viscosity}(r,\ h)\ = \frac{45}{\pi h^6} \begin{cases}(h-\|r\|) & 0 \leq \|r\| \leq 0 \\ 0 & else\end{cases}$$"> </div>
</div>


<div><h3>5 Surface Tension</h3>

<p align="justify">Surface tension forces is modelled based on proposal by Morris[4]. Molecules in a fluid are subject to attractive forces from neighboring molecules. Inside the fluid these intermolecular forces are equal in all directions and the net force is unbalanced and acts in the direction of surface normal towards fluid, minimising the surface curvature.</p>
<p align="justify">In literature, the surface is found using a field refered as color which is 1 at particle center and 0 elsewhere.</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large c_s(r) = \sum_{\forall j}\ \frac{m_j}{\rho_j}\ W(r,\ h)$" alt="$$c_s(r) = \sum_{\forall j}\ \frac{m_j}{\rho_j}\ W(r,\ h)$$"> </div>
<p align="justify">and surface normal ppointing into the fluid,</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large n\ =\ \nabla c_s$" alt="$$n\ =\ \nabla c_s$$"> </div>
<p align="justify">the divergence of n measures the curvature of the surface,</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large k = \frac{-\nabla^2 c_s}{\|n\|}$" alt="$$k = \frac{-\nabla^2 c_s}{\|n\|}$$"> </div>

<p align="justify">The minus is necessary to get positive curvature for convex fluid volumes. The force acting near the surface,</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large f_i^{surface} = \sigma k n = -\sigma\ \nabla^2 c_s \frac{n}{\|n\|}$" alt="$$f_i^{surface} = \sigma k n = -\sigma\ \nabla^2 c_s \frac{n}{\|n\|}$$"> </div>
</div>


<div><h3> 3.6 Leapfrog Integration Scheme</h3>

<p align="justify"><b>Leapfrog integration</b> is a method for numerically integrating second order differential equations particularly in the case of a dynamical system of classical mechanics. It is similar to the velocity Verlet method, which is a variant of <b>Verlet integration</b>. Leapfrog integration is equivalent to updating positions <img src="https://render.githubusercontent.com/render/math?math=$x(t)" alt="$x(t)$"> and velocities <img src="https://render.githubusercontent.com/render/math?math=$v(t)=\dot x(t)$" alt="$v(t)=\dot x(t)$"> at interleaved time points, staggered in such a way that they <b>"leapfrog"</b> over each other.
The <b>"kick-draft-kick"</b> for of Leapfrog Integration Scheme can be writen as :</p>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large v_{t+0.5\Delta t} = v_t + a_t \frac{\Delta t}{2}$" alt="$$v_{t+0.5\Delta t} = v_t + a_t \frac{\Delta t}{2}$$"> </div>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large x_{t+\Delta t} = x_t + v_{t+0.5\Delta t} \Delta t$" alt="$$x_{t+\Delta t} = x_t + v_{t+0.5\Delta t} \Delta t$$"> </div>
<div align="center"><img src="https://render.githubusercontent.com/render/math?math=$\large v_{t+0.5\Delta t} = v_t + a_{t+\Delta t} \frac{\Delta t}{2}$" alt="$$v_{t+0.5\Delta t} = v_t + a_{t+\Delta t} \frac{\Delta t}{2}$$"> </div>
</div>
</div>

## References:
[1] R. A. Gingold and J. J. Monaghan. "Smoothed particle hydrodynamics". Monthly Notices of the Royal Astronomical Society 181, pp.375-389, 1977.

[2] L. B. Lucy. "A numerical approach to the testing of the fussion hypothesis". Astrophysical Journal 82, pp. 1013-1024, 1977.

[3] Müller, Charypar and Gross, "Particle-Based Fluid Simulation for Interactive Applications". The Eurographics Association 2003.

[4] N. Foster and R. Fedkiw. Practical animation of liquids. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 23–30. ACMPress, 2001.

[5] keel, R. D., "Variable Step Size Destabilizes the Stömer/Leapfrog/Verlet Method", BIT Numerical Mathematics, Vol. 33, 1993, p. 172–175.