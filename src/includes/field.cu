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

__global__
void cal_density(particle* p, double ro_0, int N, double h){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        double mass = 0;
        for(int j=0; j<N; j++){
            if(j!=idx){
                double* r = subtract(p[idx].g_position(),p[j].g_position());
                mass += p[j].g_mass() * w_poly6(r, h);
            }
        }
        p[idx].s_density(mass);
        p[idx].update_md();
    }
}