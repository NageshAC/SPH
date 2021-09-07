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
        double y = 0;

        f << (int)(7*7*7);
        for(double z=0.04; z<=0.1; z+=0.01){ 
            for(double y=0.07; y<=0.13; y+=0.01){
                for(double x=0.04; x<=0.1; x+=0.01){
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
        f << "vtk_out_freq     " << 4 <<endl;
        f << "time_end         " << 5 <<endl;
        f << "del_t            " << 0.01 <<endl;
        f << "g                " << 0 << " " << -9.81 << " " << 0 <<endl;
        f << "ro_0             " << 1000 <<endl;
        f << "mu               " << 3.5 <<endl;
        f << "sigma            " << 0.0728 <<endl;
        f << "l                " << 7.065 <<endl;
        f << "k                " << 3 <<endl;
        f << "h                " << 0.03 <<endl;
        f << "CR                " << 0.03 <<endl;
        f << "x_min            " << 0 <<endl;
        f << "x_max            " << 0.15 <<endl;
        f << "y_min            " << 0 <<endl;
        f << "y_max            " << 0.15 <<endl;
        f << "z_min            " << 0 <<endl;
        f << "z_max            " << 0.15;

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