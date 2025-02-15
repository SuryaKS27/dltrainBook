# 1 GPU via Colab 
  
##  f1, f2,f3,f4,f5,f6,f7,f7a,f7b,fp16f7c, fp32f7c ( cu file )
    
    used in colab by using following steps
    upload f1.cu file in to colab project 

    	!nvcc f1.cu -o a1.out
     	!./a1.out
    
##  Example 1 ( cu file )
    
    use ex1.cu in Notebook cell ( copy and past and run in cell)

##  Example 2 ( cu file )
    
    use ex2.cu in Notebook cell ( copy and past and run in cell)



# 2. Python code in GPU device
 
## 2.1 Example 3 ( python  file)
    
     ( Note  ....  use following with TensrFlow version >= 2 )
     
    use ex3.py in Notebook cell ( copy and past and run in cell)

## 2.2  Example 4 ( python file )
    
     ( Note  ....  use following with TensrFlow version >= 2 )
     
    use ex4.py in Notebook cell ( copy and past and run in cell)

## 2.3  Example 5 ( python file )
    
     ( Note  ....  use following with TensrFlow version >= 2 )
     
    use ex5.py in Notebook cell ( copy and past and run in cell)


 ##  2.4 tst1.py 
  
  ( Note  ....  use following with TensrFlow version < 2 )
  
import tensorflow as tf1
import tensorflow as tf2

 Matrix Multiplicaiton is using GPU device gpu:0. 
 tf1.device('/gpu:0') is  allocating gpu:0 device to tf1 and   matrix multiplication is using  gpu:0 .  Same is shown in " c = tf1.matmul(a, b)" 

 Matrix addition is using GPU device gpu:1. 
 tf1.device('/gpu:1') is  allocating gpu:0 device to tf2 and   matrix addition is using  gpu:1 .  Same is shown in "   c1 = tf2.matadd(a1, b1)" 
 
 
 
 ##  2.5  tst3.py 
  
    ( Note  ....  use following with TensrFlow version < 2 )
    
import tensorflow as tf
import time

time_matmul(x) is a function and it is forced to  use CPU and also after that GPU is used to compute matrix multiplication.  time package is used and this is help ful to print time taken by CPU and also GPU to multiply 1000 x 1000 matrix with another 1000 x 1000 matix. 
 
 
  ##  2.6  tst4.py 
  
    ( Note  ....  use following with TensrFlow version < 2 )
    
import tensorflow as tf
import time

device in ['/gpu:2', '/gpu:3']: is using  gpu devices 2  and 3 to perform append task a 
addition is done in cpu device.

![image](https://user-images.githubusercontent.com/58679469/229200632-3468741f-802f-4a96-9015-b7dbbd8bc2f0.png)


 
# 3.CUDA  code for GPU device
  
 ##  3.1  addincpu.cpp
  g++ is used to run above given c++ file in cpu. 
 
        g++ addincpu.cpp -o add 
	
        ./add  
        
        hi...Max error: 0 


export PATH=/usr/local/cuda-10.2/bin:$PATH <br>
Above provides access to nvcc which is tool to create excutable in gpu <br>

    nvcc addincpu.cpp -o add 
    
    ./add  <br>
    
     hi...Max error: 0 <br>

 
 Above ode  runs on  CPU  and it is called as  host code
 
  
 ##  3.2  addingpu.cu  <br>
  
  CUDA code is used to run above given file in gpu device.   <br>
  nvcc is used to create executable file from .cu file
  
      nvcc addingpu.cu -o add 
      
      ./add  
      
      hi...Max error: 0 
 
  Perform profile  of given CUDA code by using  nvprof tool from NVIDIA. 
  
         nvprof ./add 
 
        ==3023== NVPROF is profiling process 3023, command: ./add 
         Max error: 0 
         ==3023== Profiling application: ./add 
          ==3023== Profiling result:  

       |Type           |Time(%) |Time     |Calls  |Avg     |Min     |  Max   |Name                     | 
       |---------------|------- |-------- |-------|--------|--------|--------|------------------------ | 
       |GPU activities:|100.00% |68.815ms |    1  |68.815ms|68.815ms|68.815ms|add(int, float*, float*) | 


  
   ![image](https://user-images.githubusercontent.com/58679469/229171538-cc2a6003-f07d-4e29-8128-603b3c0267da.png)
   
Using a multi-dimensional block means that we have to be careful about distributing this number of threads among all the dimensions. In a 1D block, we can set 1024 threads at most in the x axis, but in a 2D block, if you set 2 as the size of y, you cannot exceed 512 for the x

For example, 
dim3 threadsPerBlock(1024, 1, 1) is allowed,  <br>
as well as dim3 threadsPerBlock(512, 2, 1),  <br>
but not dim3 threadsPerBlock(256, 3, 2).<br>

  
  ## 3.2.1  add
  Kernel function add is used to perform vector addition in gpu.  
  
        add<<<1, 1>>>(N, x, y);

Now that you’ve run a kernel with one thread that does some computation, how do you make it parallel? The key is in CUDA’s <<<1, 1>>>syntax. This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU. There are two parameters here, but let’s start by changing the second one: the number of threads in a thread block. CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.


  add<<<1, 256>>>(N, x, y);  is using 1 block and its size is 256 threads   <br>
  Where N is lenght of vector x and y. These N,x,y are inputs.  <br>
  Ouput is stored in y vector ( y = x+y )
  
       __global__  
        void add(float *x, float *y) 
        { 
          int i= threadIdx.x; 
           y[i] = x[i] + y[i]; 
        }  

Following is using above add inside main. 

       add<<<1, 256>>>(x, y);
 
       nvcc addingpu.cu -o add 
      ./add  
       hi...Max error: 0 
 

Perform profile  of given CUDA code by using  nvprof tool from NVIDIA. 
    
       nvprof ./add 
       
       =3181== NVPROF is profiling process 3181, command: ./add 
        Max error: 0 
        ==3181== Profiling application: ./jaddg <br>
        ==3181== Profiling result: <br>

       |Type           |Time(%) |Time     |Calls  |Avg     |Min     |  Max   |Name                     | 
       |---------------|------- |-------- |-------|--------|--------|--------|------------------------ | 
       |GPU activities:|100.00% | 7.0370  |    1  |7.0370ms|7.0370ms|7.0370ms|add(int, float*, float*) | 


   Notr : nvprof might  have issues in some jetson xavier NX devices ( wlong with Jetpack ). 
   Following might help to over come challenges in using nvprof

   	sudo /usr/local/cuda/bin/nvprof ./<your excutable file name>

   For example

	sudo /usr/local/cuda/bin/nvprof ./add

   
  ## 3.2.2  addT
  Kernel function addT is used to perform vector addition in gpu.  <br>
  addT<<<2, 32>>>(N, x, y);  is using 2 blocks and blockSize is 32 threads   <br>
  Where N is lenght of vector x and y. These N,x,y are inputs.  <br>
  Ouput is stored in y vector ( y = x+y )
  
  
   ## 3.2.3  addBT
  Kernel function add is used to perform vector addition in gpu.  <br>
  
  ![image](https://user-images.githubusercontent.com/58679469/229173375-a1c4ba72-9d7e-4cfd-8dc1-8f2cfed69986.png)

int blockSize = 256; <br> 
int numBlocks = (N + blockSize - 1) / blockSize; <br> 
addBT<<<numBlocks, blockSize>>>(N, x, y); <br> 


  
  int blockSize = 32;  <br>
  int numBlocks = (N + blockSize - 1) / blockSize;  <br> 
  addBT<<<numBlocks, blockSize>>>(N, x, y);  is using numBlocks1 and blockSize is 32 threads  <br>
  Where N is lenght of vector x and y. These N,x,y are inputs. <br>
  Ouput is stored in y vector ( y = x+y )
  
  
  3.2.4 Matrix Add
  
      __global__  
      void matadd(float *x, float *y) 
      { 
        int i= threadIdx.x; 
        int j= threadIdx.x; 
        y[i][j] = x[i][j] + y[i][j]; 
      }   
  
 Following code inside main is making  call above to "add"  which is defined as kernel.
  
    int numBlocks = 1;  
    dim3  threadsperBlock(256,256) ; 
    matadd<<<numBlocks, threadsperBlock>>>(N, x, y); 


ThreadIdx is a 3-component vector, so that threads can be identified using a one-dimensional, two-dimensional, or three-dimensional thread index, forming a one-dimensional, two-dimensional, or three-dimensional block of threads, called a thread block

There are cu files and two header files. CMakeLists.txt is created and cmake is used.  Cmake could able to create make successfully   . but make did not run and that had put error in permission side to use cuda compiler.

 This issue need to be resolved.  Though cmake worked well ( apparently) but that did not create good make file. Thus need to find a issue in  CMakeLists.txt and  fided as well. Mentioned effort is discussed in the following slides. .bashrc fie edited and CUDA path related infra is added.  After all these, successfully  worked with

       mat/bd$ cmake  .. 
       mat/bd$ make  .. 
 
   Then executed matrix multiplication in GPU . things went well and all these dicsuccsued in upcoming slides.

CMakeLists.txt <br>
  ![image](https://user-images.githubusercontent.com/58679469/229201051-6eabe0b5-3add-425e-bdd3-2ba96631b792.png)

Following to create make file by using cmake <br>
![image](https://user-images.githubusercontent.com/58679469/229201415-923bd761-ca27-4b3d-ac61-02c854fa3c33.png)

make <br>
![image](https://user-images.githubusercontent.com/58679469/229201728-b64a8def-3f08-4e4d-9b2d-1cd81bb12379.png)

run matmul <br>
![image](https://user-images.githubusercontent.com/58679469/229201836-a971974e-7923-4ead-8718-7b493870bab3.png)


Note cmake and make are used successfully 

  3.2.5 Matrix Multiplication
  
  
  CMakeLists.txt file is given below.  Following worked for Matrix multiplication

        cmake_minimum_required(VERSION 3.10.2)
        enable_language(CUDA)   // solution uncomment this line
        #set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/")         // solution .. comment this libe
        add_executable(DLtrain ../matsrc/jkernel.cu ../matsrc/jmatMul.cu)
        set_target_properties(DLtrain PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
        set_property(TARGET DLtrain PROPERTY CUDA_STANDARD 11)


CUDA path is set in .bashrc and then commented this line and un commented earlier line.

/usr/local/cuda-10.2/  

CUDA SDK installed in this folder

Objective is to create a function to multiply a Matrix in  GPU.  But call is made from CPU with data generated in CPU. This project include, kernel.cu, kernel.h, jmatArrau.h, jmatMul.h
  
mat folter is having  CMakeLists.txt  <br>
matsrc subfolder is having soruce  code for matrix multiplicaiton in gpu
   
     /mat/bd/cmake ..
     /mat/bd/make ..
     ./bin/MatMul
     
 # 4. Handling PTX file creation for Run Time
 
 Following command line instructions, 
 
 make output “user2” from “ex1.cu” and run it as well.  <br>
 Produce the PTX for the cuda kernel <br>


	      nvcc -o ex1.ptx -ptx -arch=compute_32 ex1.cu
	      nvcc -arch=sm_32 -o jk2 ex1.cu -run
	
	      //  sm_36  and  sm_34   has n issue with nvcc. 
	      //  Thus   sm_32  is used. 

  Use [Link](https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/ ) to get tutorials on CUDA
  
  Build  Application  by  using  Make. File with  *.cu extension  is used  to build  application 
  that runs  partly in  Power  9  cpu  and  rest  in  CUDA  core  of  GPU. 
  For  GPU  nvcc  is  NVIDIA  (R)  Cuda  compiler  driver  and  its  version “Cuda  compilation  tools,  release  10.1, V 10.1.243  “

For  Host  CPU  g++  (Ubuntu  7.4.0−1ubuntu1˜18.04.1)  7.4.0 <br>
make  (  gnu  make  4.1  is  used)  is  used  to  build  applications.



  
 # 5. Handling CUDA core in GPU
 
How to run TensorFlow Object Detection model on Jetson Nano?

[Keras model in Jetson Nano](https://medium.com/swlh/how-to-run-tensorflow-object-detection-model-on-jetson-nano-8f8c6d4352e8) How to run TensorFlow Object Detection model on Jetson Nano

What are Freezing Tensorflow models? 

[CFreezing Tricks!](https://cv-tricks.com/how-to/freeze-tensorflow-models/)


About loss functions, regularization and joint losses : multinomial logistic, cross entropy, square errors, euclidian, hinge, Crammer and Singer, one versus all, squared hinge, absolute value, infogain, L1 / L2 - Frobenius / L2,1 norms, connectionist temporal classification loss Loss Functions and Optimization Algorithms. 










 
  # Reference
  1. [Click](https://developer.nvidia.com/blog/even-easier-introduction-cuda/  )  An Even Easier Introduction to CUDA 
  2. [Click](https://docs.nvidia.com/cuda/)  CUDA Toolkit Documentation  
  3. [Click](https://nichijou.co/cudaRandom/ ) CUDA C++ Best Practices Guide
  4. [Loss function in Optimization](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)
  5. [Loss functions 1](http://christopher5106.github.io/deep/learning/2016/09/16/about-loss-functions-multinomial-logistic-logarithm-cross-entropy-square-errors-euclidian-absolute-frobenius-hinge.html)
  6. [CUDA Tool Kit 12.1](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1404&target_type=clusterlocal)
  7. [Optimizing optimization algorithms ](http://news.mit.edu/2011/convexity-0715)
  8. [Optimization Tricks](http://news.mit.edu/2015/optimizing-optimization-algorithms-0121)
  9. [Keras model and Jetson Nano](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/) How to run Keras model on Jetson Nano

