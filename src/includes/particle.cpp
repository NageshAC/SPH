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
 
 class base_particle{
     protected:
         double density, position[3], velocity[3];
     public:
         base_particle(){
             this->density = 0.;
             for(int i=0;i<3; i++) this->position[i] = 0.;
             for(int i=0;i<3; i++) this->velocity[i] = 0.;
         }
 
         base_particle(const base_particle& other){
             this->density = other.density;
             for(int i=0;i<3; i++) this->position[i] = other.position[i];
             for(int i=0;i<3; i++) this->velocity[i] = other.velocity[i];
         }
 
         // __host__ __device__
         // base_particle &operator=(base_particle const &other){
         //     density = other.density;
         //     for(int i=0;i<3; i++) position[i] = other.position[i];
         //     for(int i=0;i<3; i++) velocity[i] = other.velocity[i];
         //     return *this;
         // }
 
         //*************************************************
         // setters
         //*************************************************
 
             __host__ __device__
             inline void s_density(double other){this->density = other;}
 
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
 
         //****************************************************
         // getters
         //****************************************************
 
             __host__ __device__
             inline double g_density(){return this->density;}
 
             __host__ __device__
             inline double* g_position(){return this->position;}
 
             __host__ __device__
             inline double* g_velocity(){return this->velocity;}
 };
 
 class particle : public base_particle{
     private:
         double mass, m_d, pressure, color, n[3], fT[3];
         double fP[3], fG[3], fV[3], fS[3];
         int cell[3];
     public:
         // constructor
         particle(){
             this->m_d = 0.0;
             this->pressure = 0.0;
             this->mass = 0.0;
             this->color = 0.0;
             for(int i=0;i<3; i++) this->fP[i] = 0.0;
             for(int i=0;i<3; i++) this->fG[i] = 0.0;
             for(int i=0;i<3; i++) this->fV[i] = 0.0;
             for(int i=0;i<3; i++) this->fS[i] = 0.0;
             for(int i=0;i<3; i++) this->fT[i] = 0.0;
             for(int i=0;i<3; i++) this->n[i]=0.0;
             for(int i=0;i<3; i++) this->cell[i]=0;
         }
 
         //*************************************************
         // setters
         //*************************************************
 
             __host__ __device__
             inline void s_mass(double other){this->mass = other;}
 
             __host__ __device__
             inline void s_md(double other){this->m_d = other;}
 
             __host__ __device__
             inline void s_color(double other){this->color = other;}
 
             __host__ __device__
             inline void s_pressure(double other){this->pressure = other;}
 
             __host__ __device__
             inline void s_n(double *other){
                 for(int i=0;i<3; i++)
                     this->n[i] = other[i];
             }
 
             __host__ __device__
             inline void s_fT(const double *other){
                 for(int i=0;i<3; i++)
                     this->fT[i] = other[i];
             }
 
             __host__ __device__
             inline void s_fP(const double *other){
                 for(int i=0;i<3; i++)
                     this->fP[i] = other[i];
             }
 
             __host__ __device__
             inline void s_fG(const double *other){
                 for(int i=0;i<3; i++)
                     this->fG[i] = other[i];
             }
             
             __host__ __device__
             inline void s_fV(const double *other){
                 for(int i=0;i<3; i++)
                     this->fV[i] = other[i];
             }
             
             __host__ __device__
             inline void s_fS(const double *other){
                 for(int i=0;i<3; i++)
                     this->fS[i] = other[i];
             }
         //*************************************************
         // getters
         //*************************************************
 
             __host__ __device__
             inline double g_mass(){return this->mass;}
 
             __host__ __device__
             inline double g_color(){return this->color;}
 
             __host__ __device__
             inline double g_md(){return this->m_d;}
 
             __host__ __device__
             inline double g_pressure(){return this->pressure;}
 
             __host__ __device__
             inline double* g_n(){return this->n;}
 
             __host__ __device__
             inline double* g_fT(){return this->fT;}
         
             __host__ __device__
             inline double* g_fG(){return this->fG;}
 
             __host__ __device__
             inline double* g_fV(){return this->fV;}
 
             __host__ __device__
             inline double* g_fS(){return this->fS;}
 
             __host__ __device__
             inline double* g_fP(){return this->fP;}
             
             __host__ __device__
             inline int* g_cell(){return this->cell;}
 
         //*************************************************
         // common functions
         //*************************************************
 
             __host__ __device__
             inline void update_md(){
                 if(density != 0)
                     this->m_d = mass/density;
                 else m_d = -1;
             }
         
             __host__ __device__
             inline void reset_force(){
                 for(int i=0; i<3; i++)this->fT[i] = 0.0;
                 for(int i=0;i<3; i++) this->fP[i] = 0.0;
                 for(int i=0;i<3; i++) this->fG[i] = 0.0;
                 for(int i=0;i<3; i++) this->fV[i] = 0.0;
                 for(int i=0;i<3; i++) this->fS[i] = 0.0;
                 for(int i=0; i<3; i++) this->n[i] = 0.0;
             }
         
             __host__ __device__
             inline void update_cell(double h){
                 for(int i=0; i<3; i++) cell[i] = int(position[i]/h);
             }
 
             __host__ __device__
             inline bool is_neighbour(particle* p){
                 const int* cj = p->g_cell();
 
                 if(
                     cell[0]-1 <= cj[0] && cj[0] <= cell[0]+1 &&
                     cell[1]-1 <= cj[1] && cj[1] <= cell[1]+1 &&
                     cell[2]-1 <= cj[2] && cj[2] <= cell[2]+1
                 )  return true;
                 else return false;
             }
 
             __host__ __device__
             inline bool is_neighbour(const int* cj){
                 if(
                     cell[0]-1 <= cj[0] && cj[0] <= cell[0]+1 &&
                     cell[1]-1 <= cj[1] && cj[1] <= cell[1]+1 &&
                     cell[2]-1 <= cj[2] && cj[2] <= cell[2]+1
                 )  return true;
                 else return false;
             }
         
 };