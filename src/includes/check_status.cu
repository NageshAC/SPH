/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-08-10 11:44:00
 * @modify date 2021-08-10 11:44:00
 * @desc Contains error diagnosis functions
 */

 #pragma once
 
 #include<iostream>
 #include<cstdlib>
 #include<cuda_runtime.h>
 
 //************************************************************
 // check last kernel launch for error
 //************************************************************
 static inline void cuda_check_last_kernel(
     std::string const& err
 ){
 
     /**
      * @brief Checks the status of last kernel call.
      * @param kernel_name
      * @return exit with -1
      */
 
     auto status = cudaGetLastError();
     if(status != cudaSuccess){
         std::cout << "Error: CUDA kernel launch: " << err 
             <<cudaGetErrorString(status) << std::endl;
         exit(-1);
     }
 }
 
 //************************************************************
 // Check CUDA API calls for error
 //************************************************************
 static inline void cuda_status(
     cudaError_t err, std::string message = "----", 
     std::string file = " \" this \" ", int line = -1
 ){
 
     /**
      * @brief Checks the status of API call.
      * @param err error code from API call
      * @param message message that needs to be included
      * @param file In whish file the error has occured.
      * @param line In which line the error has occured.
      * @return exit with -1
      */
 
     if(err != cudaSuccess){
         std::cout << "Error: "<<file<<" in line " << line << std::endl
             << "Message: " << message << std::endl << " CUDA API call: "
             << cudaGetErrorString(err) << std::endl;
         exit(-1);
     }
 }