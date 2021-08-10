/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-10 12:11:29
 * @modify date 2021-08-10 12:11:29
 * @desc Contains few operators definitions used in this project.
 */

 #pragma once

 #include<iostream>
 #include<cmath>
 
 #include<cuda_runtime.h>

//**************************************************************
// copy function
//**************************************************************
__device__ __host__
void copy(double* x, const double* y, int n = 3){
    for(int i=0; i<n; i++)
        x[i] = y[i];
}

//**************************************************************
// l2 norm
//**************************************************************
__device__
double norm(const double* x){
    double result = 0;
    for(int i=0; i<3; i++){
        result += pow(x[i],2);
    }
    return sqrt(result);
}

//**************************************************************
// vector const multiplication
//**************************************************************
__device__
void multiply(const double* c, double* x){
    for(auto i=0; i<3;i++) x[i] *= *c;
}

__device__
double* multiply(const double c, const double* x){
    double* r = new double [3];

    for(auto i=0; i<3;i++) r[i] = x[i] * (c);

    return r;
}

//**************************************************************
// vector vector subtraction
//**************************************************************
__device__
double* subtract(const double* x, const double* y){
    double* r = new double [3];
    for(int i=0; i<3; i++){
        r[i] = x[i] - y[i];
    }
    return r;
}