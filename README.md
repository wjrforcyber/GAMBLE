# GAMBLE
<p align="center">
   <picture>
     <img src="./image/gamble.png" width="30%" alt="gamble logo">
   </picture>
</p>

> [!NOTE]
> This is an early stage and WIP case study of a course project related to maze route. So lots of problems/decisions are simplified. 

> [!WARNING]
> 🤡 Currently the performance is quite poor. I try to make it a GPU Accelerated Maze Route with BidirectionaL Expansion, but there are not too much CUDA integration, and I do need time to write concrete unit tests.
> Even for the CPU version, the data processing section needs to be refactored. The time cost on partitioned process takes lots of time. If you are looking for a complete solution on GPU acceleration on maze route, recommend [GAMER: GPU-Accelerated Maze Routing](https://appsrv.cse.cuhk.edu.hk/~sjlin/GAMER%20TCAD2023.pdf).


## Proposed notes
The proposed method is trying to use partitioned square to boost the efficiency of maze route. The keynotes of proposed square extraction and section extraction is shown in the [slides](./slides/GAMBLE.pdf). There are lots of corner cases after partition, and not shown in the slides, if you are still interested at this point, feel free to drop me an email.

## Build
The current experiment is performed on NVIDIA GeForce RTX 3090.
You probably need to deal with `toolchain.cmake` first, I encode it with the CUDA toolset on our server. Then simply type
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -B build -S . -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
cmake --build build
cd build
```
to build the whole project. To run all test cases, type
```bash
ctest --verbose --output-on-failure
```

## Contribute
You are welcome to contribute and create issues related to the problem. I'll try my best to give solution in my free time.