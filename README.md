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

| Cases | CPU_time | CPU_cost | GPU_time | GPU_cost |
| --- | --- | --- | --- | --- |
| `case1.txt` | 5.80913s | 608800 | 0.00998s | 1345045 |
| `case2.txt` | 0.698954s | 193416 | 0.003755s | 677810 |
| `case3.txt` | 0.233073s | 135455 | 0.000927s | 249165 |
| `case4.txt` |  0.037691s | 66573 | 0.000506s | 227689 | 
| `case5.txt` | 0.014779s | 32296 | 0.000199s | 50233 |
| `case6.txt` | 0.002633s | 12972 | 0.000179s | 23539 |
| `case7.txt` | 8.5175s | 5022 | 0.010087s | 15908 |
| `case8.txt` | 0.697453s | 1969 | 0.002699s | 5649 |
| `case9.txt` | 0.331217s | 1550 | 0.001065s | 7352 |
| `case10.txt` | 0.084318s | 1019 | 0.000514s | 6460 |
| `case11.txt` | 0.008438s | 289 | 0.000223s | 1024 |
| `case12.txt` | 38.3766s | 13652 | 0.013925s | 169478 |
| `case13.txt` | 58.2931s | 17131 | 0.017707s | 297765 |


## Contribute
You are welcome to contribute and create issues related to the problem. I'll try my best to give solution in my free time.