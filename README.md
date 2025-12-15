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
For a random generated grid with no block section consideration, cost can be estimate quite fast in GPU version on a larger grid (as far as the test shows, `N = 50` is too small for GPU to show advantages), 

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
8:            5           9               17.40               20.13                5.38                0.01
8:           10          19               38.81               37.94                0.94                0.02
8:           20          39               69.02              105.04                5.96                0.05
8:          100         199              379.28              510.16                1.22                0.23
```

on a 1000 x 1000 grid, the performance is shown as below:
```bash
8: ==========================================
8: TEST: PERFORMANCE COMPARISON (PATH ONLY)
8: ==========================================
8: 
8: Performance (Grid: 1000x1000)
8: 
8: ========================================================================================================================
8:         Pins       Pairs     CPU Direct (ms)   CPU Diagonal (ms)   GPU Diagonal (ms)      Cost Calc (ms)
8: ========================================================================================================================
8:            5           9             1631.73             2287.10                7.84                0.10
8:           10          19             3355.68             3963.80               11.13                0.21
8:           20          39             6418.81             8414.77               13.86                0.42
8:          100         199            36590.14            47270.34               11.75                2.30
```

## Contribute
You are welcome to contribute and create issues related to the problem. I'll try my best to give solution in my free time.