# Mastermind

This repo contains code to play the game of Mastermind with various numbers of colors and pins using various well-known
algorithms. There are scalar, vectorized, and GPU variants of both the scoring functions and the gameplay including a
novel algorithm for playing all games at once on the GPU, implemented in CUDA and described in
[Mastermind on the GPU](docs/Mastermind_on_the_GPU.md).

I was motivated to dig into Mastermind through volunteer work with APCS high school students taught
by [Michael Miyoshi](https://github.com/MichaelTMiyoshi) at Cedarcrest High School. They are given an assignment to
write the scoring function for Mastermind, which is quite a challenge for new students to get right. I had never heard
of the game before, and I found attempting to play all games efficiently to be a pretty fun challenge myself.

This repo holds most of my work on playing all games of various sizes quickly:

- Multiple versions of the scoring function, including a vectorized version using SSE2. See the comments
  in [codeword.inl](codeword.inl) for details, and Compiler Explorer links showing the generated code.
- The following algorithms for playing all games. See [algos.hpp](algos.hpp)
  and [solver.hpp](solver.hpp) for more details.
  - Knuth
  - Most Parts
  - Expected Size
  - Entropy
- A GPU implementation using CUDA which runs all games at once. This
  has been tailored for compute capability 8.6, CUDA 11.7, running on an NVIDIA GeForce RTX 3070.
- A simple reference CPU implementation, and a more optimized CPU variant to compare with the GPU version.
- Various gameplay optimizations from Ville[2].

## Previous Versions and Easier Implementations

I originally wrote similar code in Java, since I was first motivated by the APCS students. I've kept the Java version
pretty straightforward, and it is an easier introduction to the problem than the optimized C++ code you'll find here.
See https://github.com/mikemag/CS-Education/tree/master/APCS/Mastermind

This C++ version was originally an evolution of the Java version, based on playing one game at a time. It grew to
include a GPU implementation written in Apple's Metal API for AMD GPUs, which I then ported to CUDA for Nvidia. However,
playing one game at a time isn't the best way to take full advantage of available GPU resources, so I made a large
switch to playing all games concurrently. The older version is interesting for a) the Metal impl, and b) the caching
approach to growing a gameplay strategy over time. I've kept it on
the [game_at_a_time](https://github.com/mikemag/Mastermind/tree/game_at_a_time) branch in this repo for reference.

## Docs

* [Mastermind on the GPU](docs/Mastermind_on_the_GPU.md)
* [Efficient Scoring Functions](docs/Scoring_Functions.md)
* [Packed Indices for Mastermind Scores](docs/Score_Ordinals.md)
* [Score Caching Considered Harmful](docs/Score_Cache.md)
* [Initial Guesses](docs/initial_guesses/Initial_Guesses.md)

## Results

More results and data are in the [results directory](results/).
Using the GPU for larger games is much faster, as you would expect. 5p8c is a nice example, SolverCPUFaster vs.
SolverCUDA:

|   Strategy    | Initial Guess | Max Turns | Average Turns | CPU-only (s) | GPU-only (s) |
|:-------------:|:-------------:|:---------:|:-------------:|:------------:|:------------:|
|     Knuth     |     11223     |     7     |    5.6084     |    9.9862    |    0.0413    |
|  Most Parts   |     11223     |     8     |    5.5491     |    5.4996    |    0.0408    |
| Expected Size |     11223     |     7     |    5.4997     |    9.3927    |    0.0398    |
|    Entropy    |     11223     |     7     |    5.4854     |    8.8997    |    0.0372    |

## Strategy Output

Results are post-processed into a "strategy" for playing any game of a given size. These strategies are output
as [GraphViz](https://graphviz.org/) graphs which can be rendered with any of the standard GraphViz engines.
See the [results directory](results/) for examples.

![4p 3-15c games w/ Knuth's algorithm](results/mastermind_strategy_knuth_4p.gif)

## Building and Running

This can be built two ways: CMake and XCode 13.4, on macOS 12+ and Ubuntu 22+. If you build with just CMake without the
Nvidia CUDA Toolkit installed then the GPU support is turned off. Building on Ubuntu w/ the CUDA toolkit installed turns
on the GPU support automatically.

There is a CLion workspace included in the repo, so if you use CLion you can just clone and go. This is the way I
develop 95% of the code. Works with CLion 2022.1.3 and the bundled CMake.

The XCode project works with XCode 13.4.

This isn't an easy-to-use program; there are no command line options.
Basic config is done by editing settings at the top of main.cpp. Multiple runs over different algorithms are done by
changing the config vars in main.cpp to include the specific instantiations you wish to test.
Various experiment control flags are scattered throughout the Strategy classes.
A CSV file with stats and run parameters is produced at the end.

All development and tests done (so far) on a MacBook Pro (16-inch, 2019) running macOS Monterey 12.4, and a gaming PC
running Ubuntu 22.04. The GPU kernels have been tested on an NVIDIA GeForce RTX 3070, CUDA 11.7. The program will tell
you about the GPUs on your system and which one it selected when you run. System, CPU, and GPU details are recorded in
the .csv files.

## License

Code is licensed under the MIT license except as otherwise noted.
See [LICENSE](https://github.com/mikemag/Mastermind/blob/master/LICENSE) for details.

Documents and images are copyright by [Michael Magruder](https://github.com/mikemag) and licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## References

[1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.

[2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March 2013, arXiv:
1305.1010 [cs.GT]. https://arxiv.org/abs/1305.1010

[3] Barteld Kooi, Yet another mastermind strategy. International Computer Games Association Journal, 28(1):13–20, 2005. https://www.researchgate.net/publication/30485793_Yet_another_Mastermind_strategy

  