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
double norm(const double* x, int dim = 3){
    double result = 0;
    for(int i=0; i<dim; i++){
        result += pow(x[i],2);
    }
    return sqrt(result);
}

//**************************************************************
// vector const multiplication
//**************************************************************
__device__
void axpy(const double a, const double* x, double* y, int dim =3){
    for(int i=0; i<3; i++) y[i] += a * x[i];

}

//**************************************************************
// vector const multiplication
//**************************************************************
__device__
void multiply(const double* c, double* x, int dim = 3){
    for(auto i=0; i<dim; i++) x[i] *= *c;
}

__device__
void multiply(double* r, const double c, const double* x, int dim = 3){
    for(auto i=0; i<dim; i++) r[i] = x[i] * (c);
}

//**************************************************************
// vector vector subtraction
//**************************************************************
__device__
void subtract(double* x, const double* y, int dim = 3){
    for(int i=0; i<3; i++) x[i] = x[i] - y[i];
}

__device__
void subtract(double* r, const double* x, const double* y, int dim = 3){
    for(int i=0; i<3; i++) r[i] = x[i] - y[i];
}

//**************************************************************
// vector vector addition
//**************************************************************
__device__
void add(double* x, const double* y, int dim = 3){
    for(int i=0; i<3; i++) x[i] = x[i] + y[i];
}

__device__
void add(double* r, const double* x, const double* y, int dim = 3){
    for(int i=0; i<3; i++) r[i] = x[i] + y[i];
}