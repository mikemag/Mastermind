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

Using the GPU for larger games is much faster, as you would expect. 5p8c is a nice example:

#### 5p8c

|Strategy|Initial Guess|Max Turns|GPU Mode|Average Turns|Time (s)|CPU Scores|GPU Scores|GPU Kernels
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
|First One|45678|10|CPU|5.9250|0.0315| 2,608,292 | | |
|Knuth|11234|7|CPU|5.6142|6.7618| 3,307,924,999 | | |   
| | | |Both|5.6142|0.9931| 2,518,787 | 3,466,395,648 | 2,850| 
|Most Parts|11223|8|CPU|5.5491|6.9304| 3,327,257,765 | | |   
| | | |Both|5.5491|1.3661| 2,742,071 | 3,471,933,440 | 2,795| 
|Entropy|11234|8|CPU|5.4898|10.7578| 3,225,973,291 | | |   
| | | |Both|5.4902|0.9179| 2,639,743 | 3,370,909,696 | 2,561| 
|Expected Size|11234|7|CPU|5.5038|9.4489| 3,214,838,623 | | |   
| | | |Both|5.5022|1.3006| 2,593,404 | 3,360,686,080 | 2,531| 

There is overhead to using the GPU, of course, and there are tuning points for when you use the GPU vs. falling back to
the CPU. But in general even for reasonable games like 4p6c the GPU is quite close. And you can see that for 4p7c the
GPU is already pulling ahead on some algorithms. Because of this I really haven't played with tuning the cutoff much at all.

#### 4p6c

|Strategy|Initial Guess|Max Turns|GPU Mode|Average Turns|Time (s)|CPU Scores|GPU Scores|GPU Kernels
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
|First One|3456|7|CPU|4.6211|0.0008| 55,417 | | |
|Knuth|1122|5|CPU|4.4761|0.0085| 3,237,885 | | |   
| | | |Both|4.4761|0.0142| 62,675 | 3,373,488 | 100| 
|Most Parts|1123|6|CPU|4.3735|0.0082| 3,289,320 | | |   
| | | |Both|4.3735|0.0168| 61,113 | 3,412,368 | 109| 
|Entropy|1234|6|CPU|4.4159|0.0164| 3,320,344 | | |   
| | | |Both|4.4151|0.0168| 58,489 | 3,443,472 | 106| 
|Expected Size|1123|6|CPU|4.3935|0.0163| 3,256,505 | | |   
| | | |Both|4.3951|0.0169| 62,062 | 3,346,272 | 106| 

#### 4p7c

|Strategy|Initial Guess|Max Turns|GPU Mode|Average Turns|Time (s)|CPU Scores|GPU Scores|GPU Kernels
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
|First One|4567|8|CPU|5.0675|0.0017| 114,469 | | |
|Knuth|1234|6|CPU|4.8367|0.0345| 13,577,917 | | |   
| | | |Both|4.8367|0.0351| 116,345 | 14,478,030 | 267| 
|Most Parts|1123|6|CPU|4.7430|0.0326| 13,470,924 | | |   
| | | |Both|4.7430|0.0334| 121,453 | 14,245,133 | 257| 
|Entropy|1234|6|CPU|4.7397|0.0573| 13,183,270 | | |   
| | | |Both|4.7401|0.0337| 118,016 | 13,983,424 | 237| 
|Expected Size|1234|6|CPU|4.7530|0.0459| 13,164,912 | | |   
| | | |Both|4.7505|0.0331| 116,820 | 13,875,379 | 237| 

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
The program will tell you about the GPUs on your system and which one it selected when you run:

```
GPU devices available:

GPU name: AMD Radeon Pro 5500M
Max threads per threadgroup: 1024
Max threadgroup memory length: 65536
Max buffer length: 3758096384

GPU name: Intel(R) UHD Graphics 630
Max threads per threadgroup: 1024
Max threadgroup memory length: 65536
Max buffer length: 2147483648

Using GPU: AMD Radeon Pro 5500M
``` 

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
  