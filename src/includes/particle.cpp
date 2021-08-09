/**
 * @author NageshAC
 * @email nagesh.ac.aralaguppe@fau.de
 * @create date 2021-07-27 11:46:49
 * @modify date 2021-07-27 11:46:49
 * @desc Particle info class
 */

#pragma once

#include<iostream>

#include<cuda_runtime.h>



class particle{
    private:
        double
            density,
            mass,
            m_d,
            pressure,
            position[3],
            velocity[3],
            n[3],
            force[3];

    public:

        // constructor
        particle(){
            density = 0.0;
            m_d = 0.0;
            pressure = 0.0;
            mass = 0.0;
            for(int i=0;i<3; i++){
                position[i] = 0.0;
                force[i] = 0.0;
                n[i]=0.0;
            }
        }

        //*************************************************
        // setters
        //*************************************************
        
            __host__ __device__
            inline void s_density(double other){this->density = other;}

            __host__ __device__
            inline void s_mass(double other){this->mass = other;}

            __host__ __device__
            inline void s_md(double other){this->m_d = other;}

            __host__ __device__
            inline void s_pressure(double other){this->pressure = other;}

            __host__ __device__
            inline void s_position(double *other){
                for(int i=0;i<3; i++)
                    this->position[i] = other[i];
            }

            __host__ __device__
            inline void s_position(double other, int index){
                this->position[index] = other;
            }

            __host__ __device__
            inline void s_velocity(double *other){
                for(int i=0;i<3; i++)
                    this->velocity[i] = other[i];
            }

            __host__ __device__
            inline void s_velocity(double other, int index){
                this->velocity[index] = other;
            }

            __host__ __device__
            inline void s_n(double *other){
                for(int i=0;i<3; i++)
                    this->n[i] = other[i];
            }

            __host__ __device__
            inline void s_force(const double *other){
                for(int i=0;i<3; i++)
                    this->force[i] = other[i];
            }

        

        //*************************************************
        // getters
        //*************************************************
        
            __host__ __device__
            inline double g_density(){return this->density;}

            __host__ __device__
            inline double g_mass(){return this->mass;}

            __host__ __device__
            inline double g_md(){return this->m_d;}

            __host__ __device__
            inline double g_pressure(){return this->pressure;}

            __host__ __device__
            inline double* g_position(){return this->position;}

            __host__ __device__
            inline double* g_velocity(){return this->velocity;}

            __host__ __device__
            inline double* g_n(){return this->n;}

            __host__ __device__
            inline double* g_force(){return this->force;}
        

        //*************************************************
        // common functions
        //*************************************************

            __host__ __device__
            void update_md(){
                if(density != 0)
                    this->m_d = mass/density;
                else m_d = -1;
            }
        
            __host__ __device__
            void reset_force(){
                for(int i=0; i<3; i++)
                    this->force[i] = 0.0;
            }
        
        

        
};