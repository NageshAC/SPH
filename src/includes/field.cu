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
#include"./check_boundary.cu"

//************************************************************
// density calculation
//************************************************************
__global__
void cal_density(particle* p, double ro_0, int N, double h){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        double mass = 0;
        for(int j=0; j<N; j++){
            double r[3]; 
            subtract(r,p[idx].g_position(),p[j].g_position());
            double w = w_poly6(r, h);
            if (w==0 && idx==j) {
                printf("%d w %d is zero\n",idx,j);
                printf("%lf %lf %lf \n", r[0], r[1], r[2]);
            }
            mass += p[j].g_mass() * w;
        }
        p[idx].s_density(mass);
        p[idx].update_md();
    }
}

//************************************************************
// force calculation
//************************************************************
__global__
void cal_force(
    particle* p, const double* g, 
    double ro_0, double k, int N, double h
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){

        // gravitational force
        p[idx].s_force(g);
        multiply(&ro_0, p[idx].g_force());

        // pressure
        double pr[3], r[3], c;
        for(int i=0; i<3; i++) pr[i] = 0;
        for(int j=0; j<N; j++){
            if(idx != j){
                c = -1*k*(p[idx].g_density()+p[j].g_density()-2*ro_0)*p[j].g_md()/2;
                subtract(r,p[idx].g_position(),p[j].g_position());
                grad_spiky(r, h);
                axpy(c, r, pr);
            }
        }
        add(p[idx].g_force(), pr);
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
    const double del_t, const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        double den = p[idx].g_density();
        if(den>0){

            double c = del_t/(2*den);

            axpy(c, p[idx].g_force(), p[idx].g_velocity());

            axpy(del_t, p[idx].g_velocity(), p[idx].g_position());

            // boundary condition
            check_bound(
                p[idx].g_position(), p[idx].g_velocity(), del_t,
                xmin, xmax, ymin, ymax, zmin, zmax
            );

            axpy(c, p[idx].g_force(), p[idx].g_velocity());

        }
    }
}