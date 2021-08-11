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
    double xmin, double xmax,
    double ymin, double ymax,
    double zmin, double zmax
){
    
    double R = 0.3; // co-efficient of restitution

    if(x[0] < xmin){
        x[0] = xmin + R * (v[0]*del_t - xmin + x[0]);
        v[0] *= -0.7;
    }
    if(x[1] < ymin){
        x[1] = ymin + R * (v[1]*del_t - ymin + x[1]);
        v[1] *= -0.7;
    }
    if(x[2] < zmin){
        x[2] = zmin + R * (v[2]*del_t - zmin + x[2]);
        v[2] *= -0.7;
    }

    if(x[0] > xmax){
        x[0] = xmax - R * (v[0]*del_t - x[0] + xmax);
        v[0] *= -0.7;
    }
    if(x[1] > ymax){
        x[1] = ymax - R * (v[1]*del_t - x[1] + ymax);
        v[1] *= -0.7;
    }
    if(x[2] > zmax){
        x[2] = zmax - R * (v[2]*del_t - x[2] + zmax);
        v[2] *= -0.7;
    }

}