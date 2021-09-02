/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-10 18:45:01
 * @modify date 2021-08-10 18:45:01
 * @desc contains correction for rectangular boundary condition.
 */

 #pragma once
 
 __device__
 void check_bound(
     double* x, double* v, const double del_t,
     const double xmin, const double xmax,
     const double ymin, const double ymax,
     const double zmin, const double zmax, 
     const double CR
 ){
     
     double s = 1.5e-3; // safty factor
 
     if(x[0] < xmin + s){
         x[0] = xmin + CR * (v[0]*del_t - xmin + x[0]);
         v[0] *= -CR;
     }
     if(x[1] < ymin + s){
         x[1] = ymin + CR * (v[1]*del_t - ymin + x[1]);
         v[1] *= -CR;
     }
     if(x[2] < zmin + s){
         x[2] = zmin + CR * (v[2]*del_t - zmin + x[2]);
         v[2] *= -CR;
     }
 
     if(x[0] > xmax - s){
         x[0] = xmax - CR * (v[0]*del_t - x[0] + xmax);
         v[0] *= -CR;
     }
     if(x[1] > ymax - s){
         x[1] = ymax - CR * (v[1]*del_t - x[1] + ymax);
         v[1] *= -CR;
     }
     if(x[2] > zmax - s){
         x[2] = zmax - CR * (v[2]*del_t - x[2] + zmax);
         v[2] *= -CR;
     }
 
 }
 
 __device__
 void check_collision (particle* p){
 
 }