/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-09 17:35:32
 * @modify date 2021-08-09 17:35:32
 * @desc Particle instantaneous data in vtk file format for visualisation.
 */

 #pragma once
 
 #include<iostream>
 #include<string>
 #include<fstream>
 #include<iomanip>
 
 #include "./includes/particle.cpp"
 using namespace std;
 
 void write_VTK(
     string fileName, int file_count,
     particle* p, const int& N, 
     const bool verbose = false
 ){
     fstream file;
     string ffName;
     string vtk_version = "# vtk DataFile Version 4.0",
         comments = "HESPA simulation",
         file_type = "ASCII",
         dataset = "DATASET UNSTRUCTURED_GRID";
     
     ffName = "./sln/" + fileName + "_" + to_string(file_count) + ".vtk";
     file.open(ffName, ios::out);
 
     if(file.is_open()){
 
         int dim = 3;
 
         // VTK file header
         file<<vtk_version<<endl<<comments<<endl<<file_type<<endl
             <<dataset<<endl<<"POINTS "<<N<<" double"<<endl;
 
         // X points
         for(int j=0; j<N; j++){
             double* x = p[j].g_position();
             for(int k=0; k<dim; k++)
                 file<<setprecision(6)<<std::fixed<<x[k]<<" ";
             file<<endl;
         }
 
         // density points
         file<<"CELLS 0 0\nCELL_TYPES 0\nPOINT_DATA "<<N
             <<"\nSCALARS density double\nLOOKUP_TABLE default\n";
         for(int j=0; j<N; j++)
             file<<setprecision(6)<<std::fixed<<p[j].g_density()<<endl;
 
         // for Testing purposes only
         // for(int j=0; j<N; j++){
         //     double* f = p[j].g_fT();
         //     for(int k=0; k<dim; k++)
         //         file<<setprecision(6)<<std::fixed<<f[k]<<" ";
         //     file<<endl;
         // }
 
         // V data
         file<<"VECTORS v double\n";
         for(int j=0; j<N; j++){
             double* v = p[j].g_velocity();
             for(int k=0; k<dim; k++)
                 file<<setprecision(6)<<std::fixed<<v[k]<<" ";
             file<<endl;
         }
 
         if (verbose) cout<<"\033[1;32m\n\tDone writing " << fileName + "_" + to_string(file_count) + ".vtk" << "\n\033[0m\n";
 
         file.close();
 
     }
     else{
         cout<<"\033[1;31m\n\tThe .vtk file cannot be opened.\n\033[0m\n";
         exit(202);
     }
 
 }
 