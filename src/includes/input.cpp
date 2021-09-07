/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-07-28 09:50:26
 * @modify date 2021-07-28 09:50:26
 * @desc Reads input from ./data/*.in file.
*/
 
#pragma once
#include<iostream>
#include<fstream>
#include<thrust/host_vector.h>
#include "./particle.cpp"

using namespace std;
using namespace thrust;

inline void printInput(particle* p,int &N){
    cout<<N<<endl;
    for(int j=0;j<N;++j){
        double m = p[j].g_mass();
        double* pos = p[j].g_position();
        double* vel = p[j].g_velocity();
        cout << m;
        for(auto i=0; i<3; i++) cout <<"\t" << pos[i];
        for(auto i=0; i<3; i++) cout <<"\t" << vel[i];
        cout << endl;
    }
}

inline host_vector<particle> readInput(string in_file, int &N){
    fstream f;
    f.open(in_file, ios::in);
    if(f.is_open()){
        f>>N;
        double out;
        int count = 0;
        host_vector<particle> temp_p(N);

        // cout<<"\n\nsize of temp: "<<temp_p.size();
        while (count < N)
        {
            // cout<<"\n"<<count<<endl;
            f>>out;
            // cout<<out<<"\t";
            temp_p[count].s_mass(out);
            for(int i=0;i<3;i++){
                f>>out;
                // cout<<out<<"\t";
                temp_p[count].s_position(out,i);
            }
            for(int i=0;i<3;i++){
                f>>out;
                // cout<<out<<"\t";
                temp_p[count].s_velocity(out,i);
            }
            // cout<<endl;
            count++;
        }

        cout << "\033[1;32m\n\tDone reading .in file.\n\033[0m\n";

        return temp_p;

    }
    else{
        cout<<"\033[1;31m\n\tThe .in file cannot be opened.\n\033[0m\n";
        exit(202);
    }
}