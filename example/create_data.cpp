/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-07 12:44:33
 * @modify date 2021-08-07 12:44:33
 * @desc creates a demo water.par water.in file with particle data
 */

#include<iostream>
#include<fstream>
#include<iomanip>

using namespace std;

void create_in(){

    fstream f;
    f.open("./data/water.in", ios::out);
    if(f.is_open()){

        double mass = 4E-6;
        double Vx = 0, Vy = 0, Vz = 0;

        f << (int)(20*20*20);
        for(double z=0.4; z<=0.6; z+=0.01){ 
            for(double y=0.7; y<=0.9; y+=0.01){
                for(double x=0.4; x<=0.6; x+=0.01){
                    f << endl << setprecision(6) << std::fixed << mass;
                    f << setprecision(6) << std::fixed << " " << x << " " << y << " " << z;
                    f << setprecision(1) << std::fixed << " " << Vx << " " << Vy << " " << Vz;
                }
            }
        }

        f.close();

        cout<<"\033[1;32m\n\tThe file water.in has been created.\n\033[0m\n";

    }
    else{
        cout << " \n";
        cout<<"\033[1;31m\n\tThe file water.in cannot be created.\n\033[0m\n";
        exit(-1);
    }

}

void create_par(){

    fstream f;
    f.open("./data/water.par", ios::out);
    if(f.is_open()){

        f << "part_input_file  " << "water.in" <<endl;
        f << "vtk_out_file     " << "water_" <<endl;
        f << "vtk_out_freq     " << 1 <<endl;
        f << "time_end         " << 5.0 <<endl;
        f << "del_t            " << 0.01 <<endl;
        f << "g                " << 0 << " " << -9.81 << " " << 0 <<endl;
        f << "ro_0             " << 998.29 <<endl;
        f << "mu               " << 3.5 <<endl;
        f << "sigma            " << 0.0728 <<endl;
        f << "l                " << 1E-2 <<endl;
        f << "k                " << 1 <<endl;
        f << "h                " << 0.002 <<endl;
        f << "x_min            " << 0 <<endl;
        f << "x_max            " << 1 <<endl;
        f << "y_min            " << 0 <<endl;
        f << "y_max            " << 1 <<endl;
        f << "z_min            " << 0 <<endl;
        f << "z_max            " << 1;

        f.close();

        cout<<"\033[1;32m\n\tThe file water.par has been created.\n\033[0m\n";

    }
    else{
        cout << " \n";
        cout<<"\033[1;31m\n\tThe file water.par cannot be created.\n\033[0m\n";
        exit(-1);
    }

}

int main(){

    create_in();
    create_par();
    
}