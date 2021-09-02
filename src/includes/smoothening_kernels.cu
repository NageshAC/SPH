/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-07-28 09:49:27
 * @modify date 2021-07-28 09:49:27
 * @desc contains definitions of all smoothening kernels.
 */


#pragma once

#include<cmath>
#include<cuda_runtime.h>

#include "./includes/operators.cu"

#define M_PI 3.14159265358979323846

using namespace thrust;

//**************************************************
// W_poly6, grad_poly6 and lap_poly6
//**************************************************

__device__
inline double w_poly6 (const double* r, const double POLY6, const double h){
    double n_r2 = norm2(r);
    double h2 = pow(h,2);
    double result;
    if(n_r2<=h2){
        result = pow(h2 - n_r2, 3);
    }
    else return 0;
    result *= POLY6;
    return result;
}

__device__
inline void grad_poly6(double* r, const double GPOLY6, const double h){
    double n_r2 = norm2(r);
    double h2 = pow(h,2);
    double result = 0;
    if(n_r2<=h2){
        result = pow(h2 - n_r2, 2);
        result *= GPOLY6;
    }
    multiply(&result, r);
}

__device__
inline double lap_poly6(const double* r, const double GPOLY6, const double h){
    double n_r2 = norm2(r);
    double h2 = pow(h,2);
    if(n_r2<=h2){
        double result;
        result = h2 - n_r2;
        result *= (3*h2-7*n_r2);
        return GPOLY6*result;
    }
    return 0;
}

//**************************************************
//  grad_spiky  for pressure field
//**************************************************
__device__
inline void grad_spiky(double* r, const double PV, const double h){
    double n_r = norm(r);
    if(0<=n_r && n_r<=h){
        double result = pow(h-n_r, 2) / (n_r*1000);
        result *= -PV;
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
inline double lap_viscosity(const double* r, const double PV, const double h){
    double n_r = norm(r);
    if(0<=n_r && n_r<=h){
        double result = h-n_r;
        result *= PV;
        return result;
    }
    return 0;
}
