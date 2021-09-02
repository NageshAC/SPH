/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-10 12:19:30
 * @modify date 2021-08-10 12:19:30
 * @desc Contains definitions of field calculation such as force, Pressure ...
*/

 #pragma once

 #include<cuda_runtime.h>
 #include"./particle.cpp"
 #include"./smoothening_kernels.cu"
 #include"./operators.cu"    
 #include"./check.cu"
 
 //************************************************************
 // density calculation
 //************************************************************
 __global__
 void cal_density(
     particle* p, const double ro_0, 
     const int N, const double h, const double POLY6
 ){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if(idx<N){
 
         p[idx].update_cell(h);
         __syncthreads();
 
         double den = 0;
         for(int j=0; j<N; j++){
             if(! p[idx].is_neighbour(p[j].g_cell())) continue;
             double r[3]; 
             subtract(r,p[idx].g_position(),p[j].g_position());
             den += p[j].g_mass() * w_poly6(r, POLY6, h);
         }
         p[idx].s_density(den);
         p[idx].update_md();
     }
 }
 
 //************************************************************
 // force calculation
 //************************************************************
 __global__
 void cal_force(
     particle* p, const double* g, 
     const double ro_0, const double k, const double mu,
     const double sigma, const double l, const int N, 
     const double h, const double GPOLY6, const double PV
 ){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if(idx<N){
 
         double *fT = p[idx].g_fT();
         double *xi = p[idx].g_position();
         double r[3], c;
 
         p[idx].reset_force();
 
         // gravitational force
         double *fG = p[idx].g_fG();
         p[idx].s_fG(g);
         multiply(&ro_0, fG);
         add(fT, fG);
 
         // pressure and viscosity
         {
             double *fV = p[idx].g_fV(); // viscosity
             double *fP = p[idx].g_fP();
             double del_v[3]; // viscosity
             p[idx].s_pressure(k*(p[idx].g_density()-ro_0));
             __syncthreads();
             for(int j=0; j<N; j++){
                 if(idx != j){
                     if(! p[idx].is_neighbour(p[j].g_cell())) continue;
 
 
                     subtract(r,xi,p[j].g_position());
 
                     subtract(del_v, p[j].g_velocity(), p[idx].g_velocity()); // viscosity
                     c = p[j].g_md() * lap_viscosity(r, PV, h); // viscosity
                     axpy(c, del_v, fV); // viscosity
 
                     c = -1*(p[idx].g_pressure() + p[j].g_pressure())*p[j].g_md()/2;
                     grad_spiky(r, PV, h);
                     axpy(c, r, fP);
 
                 }
             }
             axpy(mu, fV, fT);
             add(fT, fP);
         }
         __syncthreads();
 
         // surface tension
         {
             double *fS = p[idx].g_fS();
             // calculating c(i) and n(i)
             c = 0;
             double *n = p[idx].g_n();
             for(int j=0; j<N; j++){
                 if(! p[idx].is_neighbour(p[j].g_cell())) continue;
 
                 subtract(r,xi,p[j].g_position());
                 c += p[j].g_md() * lap_poly6(r, GPOLY6, h);
                 grad_poly6(r, GPOLY6, h);
                 axpy(p[j].g_md(), r, n);
             }
             p[idx].s_color(c);
 
             // calculating surdace force
             double n_n = norm(n);
             if(n_n >= l){
                 c *= -sigma/n_n;
                 axpy(c, n, fS);
             }
             add(fT, fS);
         }
         __syncthreads();
     }
 }
 
 //************************************************************
 // leap-frog scheme of integration
 //************************************************************
 __global__
 void cal_leapfrog(
     particle* p, 
     const double xmin, const double xmax,
     const double ymin, const double ymax, 
     const double zmin, const double zmax,
     const double del_t, const int N, const double CR){
 
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if(idx<N){
         double den = p[idx].g_density();
         double *fT = p[idx].g_fT();
         if(den>0){
 
             double c = del_t/(2*den);
 
             axpy(c, fT, p[idx].g_velocity());
 
             axpy(del_t, p[idx].g_velocity(), p[idx].g_position());
 
             // boundary condition
             check_bound(
                 p[idx].g_position(), p[idx].g_velocity(), del_t,
                 xmin, xmax, ymin, ymax, zmin, zmax, CR
             );
 
             axpy(c, fT, p[idx].g_velocity());
 
         }
     }
 }