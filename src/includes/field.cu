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

//************************************************************
// density calculation
//************************************************************
__global__
void cal_density(particle* p, double ro_0, int N, double h){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        double mass = 0;
        for(int j=0; j<N; j++){
            if(j!=idx){
                double r[3]; 
                subtract(r,p[idx].g_position(),p[j].g_position());
                mass += p[j].g_mass() * w_poly6(r, h);
            }
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
    particle* p, const double* g, double ro_0, int N, double h
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){

        // gravitational force
        p[idx].s_force(g);
        multiply(&ro_0, p[idx].g_force());

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

            axpy(c, p[idx].g_force(), p[idx].g_velocity());
        }
    }
}