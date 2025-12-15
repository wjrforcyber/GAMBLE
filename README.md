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
For direct case test, eg.
```bash
cd build
./loadTestCases < ../cases/case13.txt
```

## Experiment

### Initial result
Since path tracking is still using simple L shape, costs need to be upgraded.

> [!WARNING]
`case4` has some issue now.


| Cases | CPU_time | CPU_cost | GPU_time | GPU_cost |
| --- | --- | --- | --- | --- |
| `case1.txt` | 5.8171s | 608800 | 0.013s | 1341898 |
| `case2.txt` | 0.679213s | 193416 | 0.003s | 558751 |
| `case3.txt` | 0.279113s | 135455 | 0.001s | 215182 |
| `case4.txt` |  - | - | - | - | 
| `case5.txt` | 0.013782s | 32296 | 0.000s | 53898 |
| `case6.txt` | 0.002987s | 12972 | 0.000s | 27779 |
| `case7.txt` | 8.54784s | 5022 | 0.011s | 17389 |
| `case8.txt` | 0.822265s | 1969 | 0.003s | 6970 |
| `case9.txt` | 0.392851s | 1550 | 0.001s | 5379 |
| `case10.txt` | 0.081696s | 1019 | 0.000s | 2331 |
| `case11.txt` | 0.010352s | 289 | 0.000s | 1257 |
| `case12.txt` | 36.7125s | 13652 | 0.015s | 147692 |
| `case13.txt` | 55.4788s | 17131 | 0.017s | 259705 |


## Contribute
You are welcome to contribute and create issues related to the problem. I'll try my best to give solution in my free time.