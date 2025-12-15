# GAMBLE
<p align="center">
   <picture>
     <img src="./image/gamble.png" width="30%" alt="gamble logo">
   </picture>
</p>

> [!NOTE]
> This is an early stage and WIP case study of a course project related to maze route. So lots of problems/decisions are simplified. 

> [!WARNING]
If you are looking for a complete solution on GPU acceleration on maze route, recommend [GAMER: GPU-Accelerated Maze Routing](https://appsrv.cse.cuhk.edu.hk/~sjlin/GAMER%20TCAD2023.pdf).


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

## Experiment

### Initial result
For a random generated grid with no block section consideration, cost can be estimate quite fast in GPU version. 

on a 100 x 100 grid, the performance is shown as below:
```bash
8: ==========================================
8: TEST: PERFORMANCE COMPARISON (PATH ONLY)
8: ==========================================
8: 
8: Performance (Grid: 100x100)
8: 
8: ========================================================================================================================
8:         Pins       Pairs     CPU Direct (ms)   CPU Diagonal (ms)   GPU Diagonal (ms)      Cost Calc (ms)
8: ========================================================================================================================
8:            5           9               20.59               24.09                0.22                0.02
8:           10          19               46.39               45.24                0.24                0.04
8:           20          39               84.69              125.41                0.40                0.08
8:          100         199              413.26              524.11                1.10                0.32
8:          500         999             1916.18             2536.89                4.58                1.60
```

## Contribute
You are welcome to contribute and create issues related to the problem. I'll try my best to give solution in my free time.