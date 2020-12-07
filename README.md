# Mastermind

This repo contains code to play the game of Mastermind with various numbers of color and pins using various well-known 
algorithms. There are scalar, vectorized, and GPU variants of both the scoring functions and the gameplay.

I was motivated to dig into Mastermind through volunteer work with APCS high school students taught 
by [Michael Miyoshi](https://github.com/MichaelTMiyoshi) at Cedarcrest High School. They are given an assignment to write the scoring
function for Mastermind, which is quite a challenge for new students to get right. I had never heard of the game before,
and I found attempting to play all games efficiently to be a pretty fun challenge myself.

This repo holds most of my work on playing all games of various sizes quickly:

- Multiple versions of the scoring function, including a vectorized version using SSE2. See the comments in [codeword.cpp](codeword.cpp)
for details, and Compiler Explorer links showing the generated code.
- The following algorithms for playing all games. See [strategy.hpp](strategy.hpp) and [subsetting_strategies.hpp](subsetting_strategies.hpp) for more details.
  - First consistent
  - Random consistent
  - Knuth
  - Most Parts
  - Expected Size
  - Entropy
 - A GPU implementation which runs the scoring function and the inner loops of Knuth, Most Parts, etc.
   - This is implemented using Apple's Metal API and is only slightly tailored to the AMD Radeon Pro 5500M in my MacBook Pro.
 - Various gameplay optimizations from Ville[2].

## Results

More results and data are in the [results directory](results/). 
Using the GPU for larger games is much faster, as you would expect. 5p8c is a nice example:

|Strategy|Initial Guess|Max Turns|GPU Mode|Average Turns|Time (s)|CPU Scores|GPU Scores|GPU Kernels
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
|First One|87654|9|CPU|5.9092|0.0393|2,596,79|
|Knuth|11223|7|CPU|5.6084|6.6558|3,296,841,082|
| | | |Both|5.6084|1.0228| 2,541,260.00|3,455,090,688|2,824|
|Most Parts|11223|8|CPU|5.5491|6.7347|3,327,257,765|
| | | |Both|5.5491|1.2146| 2,742,071.00|3,471,933,440|2,795|
|Entropy|11223|7|CPU|5.4846|10.6226|3,212,855,854|
| | |7|Both|5.4855|1.1068| 2,671,408.00|3,359,539,200|2,523|
|Expected Size|11223|7|CPU|5.5016|8.4684|3,218,686,624|
| | | |Both|5.4995|0.9620| 2,624,778.00|3,358,359,552|2,550|

## Docs

* [Packed Indices for Mastermind Scores](docs/Score_Ordinals.md)
* [Score Caching Considered Harmful](docs/Score_Cache.md)
* [Initial Guesses](docs/initial_guesses/Initial_Guesses.md)

*TODO: I'll be adding more details on the tradeoffs in various parts of the implementation, the GPU algorithms, performance work,
etc. over time.*

## Easier Implementations

I originally wrote similar code in Java, since I was first motivated by the APCS students. I've kept the Java version
pretty straightforward, and it is an easier introduction to the problem than the optimized C++ code you'll find here.
See https://github.com/mikemag/CS-Education/tree/master/APCS/Mastermind

## Building and Running

This can be built two ways: CMake and XCode. If you build with just CMake then the GPU support is turned off, since I
didn't feel like putting in the effort to make a proper cmake that includes building the Objective-C and Metal code
correctly. If you build with XCode then the GPU support is turned on and works automatically.

There is a CLion workspace included in the repo, so if you use CLion you can just clone and go. This is the way I 
develop 95% of the code and the only thing missing is the GPU acceleration. Works with CLion 2020.2 and the bundled CMake.

The XCode project works with XCode 12.2.

This isn't an easy-to-use command line program; there are no command line options.
Basic config is done by editing settings at the top of main.cpp. Multiple runs over different algorithms are done by changing
the tables in main.cpp to include the specific instantiations you wish to test.
Various experiment control flags are scattered throughout the Strategy classes.
A CSV file with stats and run parameters is produced at the end.

All development and tests done (so far) on a MacBook Pro (16-inch, 2019) running macOS Catalina 10.15.7. The GPU kernels
have been tested on both GPUs in my laptop: the integrated Intel UHD Graphics 630 and the discrete AMD Radeon Pro 5500M.
The program will tell you about the GPUs on your system and which one it selected when you run.

## License

Code is licensed under the MIT license except as otherwise noted.
See [LICENSE](https://github.com/mikemag/Mastermind/blob/master/LICENSE) for details.

Documents and images are copyright by [Michael Magruder](https://github.com/mikemag) and licensed under a 
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## References

[1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.

[2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March 2013, arXiv:1305.1010 [cs.GT]. https://arxiv.org/abs/1305.1010

[3] Barteld Kooi, Yet another mastermind strategy. International Computer Games Association Journal, 28(1):13–20, 2005. https://www.researchgate.net/publication/30485793_Yet_another_Mastermind_strategy

[4] Stuart Reges and Marty Stepp, Building Java Programs: a back to basics approach, 2nd edition, 2011, Addison-Wesley. https://www.buildingjavaprograms.com/
  