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

### Run bash for all 13 test cases
There's a bash script in the source,  if you run cmake it will be automatically copied to `build` directory. 
```bash
cd build
chmod +x run_test.sh
./run_test.sh
```
will run all the actually test cases in `cases` folder.


## Experiment

### Initial result
Since path tracking is still using simple L shape, costs need to be upgraded.

| Cases | CPU_time | CPU_cost | GPU_time | GPU_cost |
| --- | --- | --- | --- | --- |
| `case1.txt` | 5.18397s | 608800 | 0.009971s | 1343503 |
| `case2.txt` | 0.660135s | 193416 | 0.004082s | 571414 |
| `case3.txt` | 0.272663s | 135455 | 0.001102s | 249165 |
| `case4.txt` |  0.039683s | 66573 | 0.000554s | 179700 | 
| `case5.txt` | 0.016163s | 32296 | 0.000217s | 49517 |
| `case6.txt` | 0.002291s | 12972 | 0.000203s | 26999 |
| `case7.txt` | 6.80957s | 5022 | 0.011538s | 15908 |
| `case8.txt` | 0.7789s | 1969 | 0.003403s| 5649 |
| `case9.txt` | 0.382095s | 1550 | 0.001323s | 6196 |
| `case10.txt` | 0.085585s | 1019 | 0.000558s | 3547 |
| `case11.txt` | 0.008722s | 289 | 0.000232s | 1024 |
| `case12.txt` | 25.7014s | 13652 | 0.012315s | 135160 |
| `case13.txt` | 39.6136s | 17131 | 0.0136s | 275196 |


## Contribute
You are welcome to contribute and create issues related to the problem. I'll try my best to give solution in my free time.