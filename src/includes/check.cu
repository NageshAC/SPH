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
    
    double R = 0.4; // co-efficient of restitution
    double s = 0.01; // safty factor

    if(x[0] < xmin + s*(xmax-xmin)){
        x[0] = xmin + R * (v[0]*del_t - xmin + x[0]);
        v[0] *= -R;
    }
    if(x[1] < ymin + s*(ymax-ymin)){
        x[1] = ymin + R * (v[1]*del_t - ymin + x[1]);
        v[1] *= -R;
    }
    if(x[2] < zmin + s*(zmax-zmin)){
        x[2] = zmin + R * (v[2]*del_t - zmin + x[2]);
        v[2] *= -R;
    }

    if(x[0] > xmax - s*(xmax-xmin)){
        x[0] = xmax - R * (v[0]*del_t - x[0] + xmax);
        v[0] *= -R;
    }
    if(x[1] > ymax - s*(ymax-ymin)){
        x[1] = ymax - R * (v[1]*del_t - x[1] + ymax);
        v[1] *= -R;
    }
    if(x[2] > zmax - s*(zmax-zmin)){
        x[2] = zmax - R * (v[2]*del_t - x[2] + zmax);
        v[2] *= -R;
    }

}

__device__
void check_collision (particle* p){

}