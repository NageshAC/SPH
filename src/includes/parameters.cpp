/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-09 22:19:21
 * @modify date 2021-08-09 22:19:21
 * @desc Reads setup parameters from ./data/*.par file.
 */

#pragma once
#include<iostream>
#include<fstream>
#include<conio.h>

using namespace std;


inline void readParam(
    string file,
    string& part_in_file, int& vtk_out_freq,
    string& vtk_out_file, double& time_end,
    double& del_t, double* g, double& ro_0, 
    double& mu, double& CR,
    double& sigma, double& l, 
    double& k, double& h,
    double& x_min, double& x_max,
    double& y_min, double& y_max,
    double& z_min, double& z_max
){
    fstream f;
    f.open(file, ios::in);
    if(f.is_open()){
        
        string out;

        while(!f.eof()){
            f >> out;
            if(out == "part_input_file") f>>part_in_file;
            if(out == "vtk_out_freq") f>>vtk_out_freq;
            if(out == "vtk_out_file") f>>vtk_out_file;
            if(out == "time_end") f>>time_end;
            if(out == "del_t") f>>del_t;
            if(out == "g") for(int i=0;i<3;i++) f>>g[i];
            if(out == "ro_0") f>>ro_0;
            if(out == "mu") f>>mu;
            if(out == "sigma") f>>sigma;
            if(out == "l") f>>l;
            if(out == "k") f>>k;
            if(out == "h") f>>h;
            if(out == "CR") f>>CR;
            if(out == "x_min") f>>x_min;
            if(out == "x_max") f>>x_max;
            if(out == "y_min") f>>y_min;
            if(out == "y_max") f>>y_max;
            if(out == "z_min") f>>z_min;
            if(out == "z_max") f>>z_max;
        }
        cout<<"\033[1;32m\n\tDone reading .par file.\n\033[0m\n";
    }
    else{
        cout<<"\033[1;31m\n\tThe .par file cannot be opened.\n\033[0m\n";
        exit(202);
    }
}

inline void printParam(
    string& part_in_file, int& vtk_out_freq,
    string& vtk_out_file, double& time_end,
    double& del_t, double* g, double& ro_0, 
    double& mu,
    double& sigma, double& l, 
    double& k, double& h,
    double& x_min, double& x_max,
    double& y_min, double& y_max,
    double& z_min, double& z_max
){
    cout<<"part_input_file   "<<part_in_file<<endl;
    cout<<"vtk_out_freq     "<<vtk_out_freq<<endl;
    cout<<"vtk_out_file      "<<vtk_out_file<<endl;
    cout<<"time_end          "<<time_end<<endl;
    cout<<"del_t             "<<del_t<<endl;
    cout<<"g                 "<<g[0]<<" "<<g[1]<<" "<<g[2]<<endl;
    cout<<"ro_0              "<<ro_0<<endl;
    cout<<"mu                "<<mu<<endl;
    cout<<"sigma             "<<sigma<<endl;
    cout<<"l                 "<<l<<endl;
    cout<<"k                 "<<k<<endl;
    cout<<"h                 "<<h<<endl;
    cout<<"x_min             "<<x_min<<endl;
    cout<<"x_max             "<<x_max<<endl;
    cout<<"y_min             "<<y_min<<endl;
    cout<<"y_max             "<<y_max<<endl;
    cout<<"z_min             "<<z_min<<endl;
    cout<<"z_max             "<<z_max<<endl;
}