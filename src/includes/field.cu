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
void cal_density(particle* p, double ro_0, int N, double h){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        double den = 0;
        for(int j=0; j<N; j++){
            double r[3]; 
            subtract(r,p[idx].g_position(),p[j].g_position());
            den += p[j].g_mass() * w_poly6(r, h);
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
    double ro_0, double k, double mu,
    double sigma, double l, int N, double h
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){

        double *fi = p[idx].g_force();
        double *xi = p[idx].g_position();
        double roi = p[idx].g_density();

        // p[idx].reset_force();

        // gravitational force
        p[idx].s_force(g);  // reset force
        multiply(&ro_0, fi);
        __syncthreads();

        // pressure
        {
            p[idx].s_pressure(k*(p[idx].g_density()-ro_0));
            __syncthreads();
            double pr[3], r[3], c;
            for(int i=0; i<3; i++) pr[i] = 0;
            for(int j=0; j<N; j++){
                if(idx != j){
                    subtract(r,xi,p[j].g_position());
                    if(norm(r) <= h){
                        c = -1*(p[idx].g_pressure() + p[j].g_pressure())*p[j].g_md()/2;
                        grad_spiky(r, h);
                        axpy(c, r, pr);
                    }
                }
            }
            // printf("P(%d) = %lf %lf %lf\n", idx, pr[0], pr[1], pr[2]);
            add(fi, pr);
        }
        __syncthreads();

        // viscosity
        {
            double vs[3], del_v[3], r[3], c;
            for(int i=0; i<3; i++) vs[i] = 0;
            for(int j=0; j<N; j++){
                if(idx != j){
                    subtract(r,xi,p[j].g_position());
                    subtract(del_v, p[j].g_velocity(), p[idx].g_velocity());
                    // if(idx==0 && j==1) printf("lap = %lf md = %lf\n", lap_viscosity(r, h),p[j].g_md());
                    c = p[j].g_md() * lap_viscosity(r, h);
                    axpy(c, del_v, vs);
                    // if(idx==0 && j==1) printf("c = %lf vs = %lf %lf  %lf\n", c, vs[0], vs[1], vs[2]);
                }
            }
            axpy(mu, vs, fi);
        }
        __syncthreads();


        // surface tension
        {
            // calculating c(i)
            double color = 0;
            double r[3];
            for(int j=0; j<N; j++){
                // if(idx != j){ 
                    subtract(r,p[idx].g_position(),p[j].g_position());
                    color += p[j].g_md() * lap_poly6(r, h);
                // }
            }
            p[idx].s_color(color);

            // calculating n(i)
            double n[3];
            for(int i=0; i<3; i++) n[i] = 0;
            for(int j=0; j<N; j++){
                subtract(r,p[idx].g_position(),p[j].g_position());
                grad_poly6(r,h);
                axpy(p[j].g_md(), r, n);
            }
            p[idx].s_n(n);

            // calculating surdace force
            double n_n = norm(p[idx].g_n());
            if(n_n >= l){
                color *= -sigma/n_n;
                axpy(color, n, fi);
            }
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
    const double del_t, const int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        double den = p[idx].g_density();
        double *fi = p[idx].g_force();
        if(den>0){

            double c = del_t/(2*den);

            axpy(c, fi, p[idx].g_velocity());

            axpy(del_t, p[idx].g_velocity(), p[idx].g_position());

            // boundary condition
            check_bound(
                p[idx].g_position(), p[idx].g_velocity(), del_t,
                xmin, xmax, ymin, ymax, zmin, zmax
            );

            axpy(c, fi, p[idx].g_velocity());

        }
    }
}