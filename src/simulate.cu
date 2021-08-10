/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-09 21:21:59
 * @modify date 2021-08-09 21:21:59
 * @desc main file: contains all the function calls and kernel calls
 */

 #include<iostream>
 #include<string>

 #include<cuda_runtime.h>
 #include<thrust/host_vector.h>
 #include<thrust/device_vector.h>

 #include "./includes/check_status.cu"
 #include "./includes/parameters.cpp"
 #include "./includes/particle.cpp"
 #include "./includes/input.cpp"
 #include "./includes/wireframe.cpp"
 #include "./includes/vtk.cpp"
 #include "./includes/field.cu"

 using namespace std;
 using namespace thrust;

 int main(){

    string input_path = "./data/";
    string output_path = "./sln/";
    string par_file = "water.par";

    string full_file = input_path + par_file;

    string part_in_file, vtk_out_file;
    int vtk_out_freq;
    double time_end, del_t, ro_0, mu, sigma, l, 
        k, h, x_min, x_max, y_min, y_max, z_min, z_max;
    host_vector<double> g(3,0);

    //**************************************************
    // reading .par file
    //**************************************************
    {
        readParam(
            full_file,
            part_in_file, vtk_out_freq,
            vtk_out_file, time_end, del_t, 
            raw_pointer_cast(&g[0]), 
            ro_0, mu,
            sigma, l, k, h,
            x_min, x_max,
            y_min, y_max,
            z_min, z_max
        );
        // printParam(
        //     part_in_file, vtk_out_freq,
        //     vtk_out_file, time_end, del_t, 
        //     raw_pointer_cast(&g[0]), 
        //     ro_0, mu,
        //     sigma, l, k, h,
        //     x_min, x_max,
        //     y_min, y_max,
        //     z_min, z_max
        // );
    }

    host_vector<particle> p;
    int N, frames = (time_end/del_t);

    //**************************************************
    // reading .in files
    //**************************************************
    {
        full_file = input_path + part_in_file;
        p = readInput(full_file,N);
        // printInput(
        //     raw_pointer_cast(&p[0]),
        //     N
        // );
    }

    //**************************************************
    // boundary 
    //**************************************************
    create_wireframe(
        x_min, x_max, 
        y_min, y_max, 
        z_min, z_max
    );

    // write_VTK(vtk_out_file, 0, raw_pointer_cast(&p[0]), N);

    //**************************************************
    // CUDA Programming
    //**************************************************
    {
        // creating copy of particle vector in GPU
        device_vector<particle> d_p(p);
        device_vector<double> d_g(g);

        // creating block and grids
        cudaDeviceProp deviceProp;
        cuda_status(
            cudaGetDeviceProperties(&deviceProp,0),
            "Get device Properties",
            __FILE__, __LINE__
        );
        // cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<endl;

        int blockSize, gridSize;
        if(N<deviceProp.maxThreadsPerBlock){
            blockSize = N;
            gridSize = 1;
        }
        else{
            blockSize = deviceProp.maxThreadsPerBlock;
            gridSize = (N/deviceProp.maxThreadsPerBlock)+1;
        }

        // cout<<"Block Size: "<<blockSize<<"\nGrid size: "<<gridSize<<endl
        // <<"Frames: "<<frames<<endl;

        for(int frame = 1; frame<=frames; frame++){
            // calculate density
            cal_density<<<gridSize,blockSize>>>(
                raw_pointer_cast(&d_p[0]),
                ro_0, N, h
            );
            cudaDeviceSynchronize();

            if(frame == 1){
                p = d_p;
                write_VTK(vtk_out_file, 0, raw_pointer_cast(&p[0]), N);
            }
            
            // calculate force
            cal_force<<<gridSize,blockSize>>>(
                raw_pointer_cast(&d_p[0]),
                raw_pointer_cast(&d_g[0]),
                ro_0, N, h
            );
            cudaDeviceSynchronize();

            // leap-frog scheme of integration
            cal_leapfrog<<<gridSize,blockSize>>>(
                raw_pointer_cast(&d_p[0]),
                x_min, x_max, y_min, 
                y_max, z_min, z_max,
                del_t, N
            );
            cudaDeviceSynchronize();
            // cuda_status(cudaMemcpy(raw_pointer_cast(&p[0]), raw_pointer_cast(&d_p[0]), sizeof(particle), cudaMemcpyDeviceToHost));
            p = d_p;
            write_VTK(vtk_out_file, frame, raw_pointer_cast(&p[0]), N);

        }

    }


    return 0;
 }