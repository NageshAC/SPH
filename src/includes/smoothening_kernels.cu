/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-07-28 09:49:27
 * @modify date 2021-07-28 09:49:27
 * @desc contains definitions of all smoothening kernels.
 */


#pragma once

#include<cmath>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>

#include "./includes/particle.cpp"
#include "./includes/operators.cu"

#define M_PI 3.14159265358979323846

using namespace thrust;

//**************************************************
// W_poly6, grad_poly6 and lap_poly6
//**************************************************
__device__
inline double w_poly6 (const double* r, const double h){
    double n_r = norm(r);
    double result;
    if(0<=n_r && n_r<=h){
        result = pow(pow(h,2) - pow(n_r,2), 3);
    }
    else if(n_r > h){
        return 0;
    }
    double c = 315/(64*M_PI*pow(h,9));
    result *= c;
    return result;
    
}

__device__
inline void grad_poly6(double* r, const double h){
    double n_r = norm(r);
    if(0<=n_r && n_r<=h){
        double c = -945/(32*M_PI*pow(h,9));
        double result;
        result = pow(pow(h,2) - pow(n_r,2), 2);
        result *= c;
        multiply(&result, r);
    }
}

__device__
inline double lap_poly6(const double* r, const double h){
    double n_r = norm(r);
    if(0<=n_r && n_r<=h){
        double c = -945/(32*M_PI*pow(h,9));
        double result;
        result = pow(h,2) - pow(n_r,2);
        result *= 3*pow(h,2)-7*pow(n_r,2);
        return c*result;
    }
    return 0;
}

//**************************************************
//  grad_spiky  for pressure field
//**************************************************
__device__
inline void grad_spiky(double* r, const double h){
    double n_r = norm(r);
    if(0<n_r && n_r<=h){
        double c = -45/(M_PI*pow(h,6)*n_r);
        double result = pow(h-n_r, 2);
        result *= c;
        multiply(&result, r);
    }
    else{
        for(int i=0; i<3; i++) r[i] = 0;
    }
}

//**************************************************
//  lap_viscosity  for viscosity field
//**************************************************
__device__
inline double lap_viscosity(const double* r, const double h){
    double n_r = norm(r);
    if(0.05<=n_r && n_r<h){
        double c = 45/(M_PI*pow(h,6));
        double result = h-n_r;
        result *= c;
        return result;
    }
    return 0;
}