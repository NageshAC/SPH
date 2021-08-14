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

        double mass = 0.02;
        double Vx = 0, Vy = 0, Vz = 0;

        f << (double)(5*5*5);
        for(double z=4; z<=6; z+=0.5){ 
            for(double y=7; y<=9; y+=0.5){
                for(double x=4; x<=6; x+=0.5){
                    f << endl << setprecision(3) << mass;
                    f << setprecision(1) << std::fixed << " " << x << " " << y << " " << z;
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
        f << "time_end         " << 10.0 <<endl;
        f << "del_t            " << 0.01 <<endl;
        f << "g                " << 0 << " " << -9.81 << " " << 0 <<endl;
        f << "ro_0             " << 0.02 <<endl;
        f << "mu               " << 3.5 <<endl;
        f << "sigma            " << 0.0728 <<endl;
        f << "l                " << 1E-2 <<endl;
        f << "k                " << 2 <<endl;
        f << "h                " << 1 <<endl;
        f << "x_min            " << 0 <<endl;
        f << "x_max            " << 10 <<endl;
        f << "y_min            " << 0 <<endl;
        f << "y_max            " << 10 <<endl;
        f << "z_min            " << 0 <<endl;
        f << "z_max            " << 10;

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