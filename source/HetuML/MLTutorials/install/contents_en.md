- ## HetuML Installation Guide

  Building HetuML from source code consists the following steps:

  ### Clone the code

  Users need to clone the code from GitHub:

  ```
  git clone https://github.com/ccchengff/HetuML.git
  ```

  ### Enviroment Requirements

  - CMake >= 3.11
    - If you haven't install CMake meet this requirement,  you could select the corresponding version and download CMake from its Website at: https://cmake.org/download/. 
  - Protobuf >= 3.0.0
    - If you haven't install Protobuf meet this requirement,  you could select the corresponding version and download Protobuf from its Website at: https://github.com/protocolbuffers/protobuf/releases. 
  - OpenMP
    - If you haven't install OpenMP meet this requirement,  you could select the corresponding version and download OpenMP from its Website at: https://www.open-mpi.org/software/ompi/. 

  ### Build from Source Code

  Users need to execute the following command in the source code folder:

  ```
  cp cmake/config.cmake.template cmake/config.cmake
  ```

  Then, you could compile HetuML:

  ```
  mkdir build && cd build
  cmake ..
  make -j 8
  ```

  If the building is successfull, type this command：

  ```
  source env.exp
  ```

  ### Work with HetuML

  You can work with HetuML now! 

  You could quickly try out HetuML on a small demo in the quick start tutorial. 