// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda_runtime.h>

#include <algorithm>
#include <cuda/std/cstdint>
#include <iostream>
#include <string>

#include "codeword.hpp"
#include "compute_kernel_constants.h"
#include "cuda_gpu_interface.hpp"

using namespace std;

// The core of Knuth's Mastermind algorithm, and others, as CUDA compute kernels.
//
// Scores here are not the classic combination of black hits and white hits. A score's ordinal is (b(p + 1) - ((b -
// 1)b) / 2) + w. See docs/Score_Ordinals.md for details. By using the score's ordinal we can have densely packed set
// of counters to form the subset counts as we go. These scores never escape the GPU, so it doesn't matter that they
// don't match any other forms of scores in the rest of the program.

// Mastermind scoring function
//
// This mirrors the scalar version very closely. It's the full counting method from Knuth, plus some fun bit twiddling
// hacks and SWAR action. This is O(1) using warp SIMD intrinsics
//
// Find black hits with xor, which leaves zero nibbles on matches, then count the zeros in the result. This is a
// variation on determining if a word has a zero byte from https://graphics.stanford.edu/~seander/bithacks.html. This
// part ends with using the GPU's SIMD popcount() to count the zero nibbles.
//
// Next, color counts come from the parallel buffer, and we can run over them and add up total hits, per Knuth[1], by
// aggregating min color counts between the secret and guess.
//
// Templated w/ pinCount and explicitly specialized below. We look up the correct version to use by name during startup.

// https://godbolt.org/z/ea7YjEPqf

static __inline__ __host__ __device__ uchar4 make_uchar4(unsigned int i) { return *reinterpret_cast<uchar4 *>(&i); }

template <uint pinCount>
__device__ uint scoreCodewords(const uint32_t secret, const uint4 secretColors, const uint32_t guess,
                               const uint4 guessColors) {
  constexpr uint unusedPinsMask = 0xFFFFFFFFu & ~((1lu << pinCount * 4u) - 1);
  uint v = secret ^ guess;  // Matched pins are now 0.
  v |= unusedPinsMask;      // Ensure that any unused pin positions are non-zero.
  uint r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint b = __popc(r);

  uint mins1 = __vminu4(secretColors.x, guessColors.x);
  uint mins2 = __vminu4(secretColors.y, guessColors.y);
  uint mins3 = __vminu4(secretColors.z, guessColors.z);
  uint mins4 = __vminu4(secretColors.w, guessColors.w);
  uchar4 totals = make_uchar4(__vadd4(mins1, mins2) + __vadd4(mins3, mins4));
  uint allHits = (totals.x + totals.y) + (totals.z + totals.w);

  // Given w = ah - b, simplify to i = bp - ((b - 1)b) / 2) + ah. I wonder if the compiler noticed that.
  // https://godbolt.org/z/ab5vTn -- gcc 10.2 notices and simplifies, clang 11.0.0 misses it.
  return b * pinCount - ((b - 1) * b) / 2 + allHits;
}

// The common portion of the kernels which scores all possible solutions against a given guess and computes subset
// sizes and whether or not the guess is a possible solution.
template <uint32_t pinCount>
__device__ bool computeSubsetSizes(int totalScores, uint32_t *scoreCounts, const uint32_t secret,
                                   const uint4 secretColors, const uint32_t possibleSolutionsCount,
                                   const uint32_t *possibleSolutions, const uint4 *possibleSolutionsColors) {
  bool isPossibleSolution = false;
  for (int i = 0; i < totalScores; i++) scoreCounts[i] = 0;

  for (uint32_t i = 0; i < possibleSolutionsCount; i++) {
    uint score = scoreCodewords<pinCount>(secret, secretColors, possibleSolutions[i], possibleSolutionsColors[i]);
    scoreCounts[score]++;
    if (score == totalScores - 1) {  // The last score is the winning score
      isPossibleSolution = true;
    }
  }
  return isPossibleSolution;
}

// This takes two sets of codewords: the "all codewords" set, which is every possible codeword, and the "possible
// solutions" set. They're separated into two buffers each: one for codewords, which are packed into 32 bits, and
// one for pre-computed color counts, packed into 128 bits as 16 8-bit counters.
//
// The all codewords set is placed into GPU memory once at program start and remains constant.
//
// The possible solutions set changes each time, both content and length, but reuses the same buffers.
//
// Output is two arrays with elements for each of the all codewords set: one with the score, and one bool
// for whether the codeword is a possible solution.
//
// Finally, there's shared block memory for each thread with enough room for all the intermediate subset sizes.

extern __shared__ uint32_t tgScoreCounts[];

template <uint32_t pinCount, Algo algo>
__global__ void subsettingAlgosKernel(const uint32_t *allCodewords, const uint4 *allCodewordsColors,
                                      uint32_t possibleSolutionsCount, const uint32_t *possibleSolutions,
                                      const uint4 *possibleSolutionsColors, uint32_t *scores,
                                      bool *remainingIsPossibleSolution, uint32_t *fullyDiscriminatingCodewords) {
  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  constexpr int totalScores = ((pinCount * (pinCount + 3)) / 2) + 1;
  uint32_t *scoreCounts = &tgScoreCounts[threadIdx.x * totalScores];

  bool isPossibleSolution =
      computeSubsetSizes<pinCount>(totalScores, scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                   possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);
  remainingIsPossibleSolution[tidGrid] = isPossibleSolution;

  uint32_t largestSubsetSize = 0;
  uint32_t totalUsedSubsets = 0;
  float entropySum = 0.0;
  float expectedSize = 0.0;
  for (int i = 0; i < totalScores; i++) {
    if (scoreCounts[i] > 0) {
      totalUsedSubsets++;
      switch (algo) {
        case Knuth:
          largestSubsetSize = max(largestSubsetSize, scoreCounts[i]);
          break;
        case MostParts:
          // Already done
          break;
        case ExpectedSize:
          expectedSize += ((float)scoreCounts[i] * (float)scoreCounts[i]) / possibleSolutionsCount;
          break;
        case Entropy:
          float pi = (float)scoreCounts[i] / possibleSolutionsCount;
          entropySum -= pi * log(pi);
          break;
      }
    }
  }
  switch (algo) {
    case Knuth:
      scores[tidGrid] = possibleSolutionsCount - largestSubsetSize;
      break;
    case MostParts:
      scores[tidGrid] = totalUsedSubsets;
      break;
    case ExpectedSize:
      scores[tidGrid] = (uint32_t)round(expectedSize * 1'000'000.0) * -1;  // 9 digits of precision
      break;
    case Entropy:
      scores[tidGrid] = round(entropySum * 1'000'000'000.0);  // 9 digits of precision
      break;
  }

  // If we find some guesses which are fully discriminating, we want to pick the first one lexically to play. tidGrid is
  // the same as the ordinal for each member of allCodewords, so we can simply take the min tidGrid. We'll have every
  // member of each warp that finds such a solution vote on the minimum, and have the first of them write the
  // result. I could do a further reduction to a value per block, or a final single value, but for now I'll just
  // let the CPU take the first non-zero result and run with it.
  if (totalUsedSubsets == possibleSolutionsCount) {
    uint d = __reduce_min_sync(__activemask(), tidGrid);
    if (tidGrid == d) {
      fullyDiscriminatingCodewords[tidGrid / 32] = d;
    }
  }
}

template <uint8_t p, uint8_t c, Algo a, bool l>
CUDAGPUInterface<p, c, a, l>::CUDAGPUInterface() {
  // mmmfixme: print GPU info and which one we've selected.
  cout << "CUDA!" << endl;

  const uint64_t totalCodewords = Codeword<p, c>::totalCodewords;
  threadsPerBlock = std::min(128lu, totalCodewords);
  numBlocks = (totalCodewords + threadsPerBlock - 1) / threadsPerBlock;  // nb: round up!
  uint32_t roundedTotalCodewords = numBlocks * threadsPerBlock;

  // NB: matches the def in the compute kernel.
  const int totalScores = ((p * (p + 3)) / 2) + 1;
  sharedMemSize = sizeof(uint32_t) * totalScores * threadsPerBlock;

  auto mallocManaged = [](auto devPtr, auto size) {
    cudaError_t err = cudaMallocManaged(devPtr, size);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate managed memory (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  };

  mallocManaged((void **)&dAllCodewords, sizeof(*dAllCodewords) * roundedTotalCodewords);
  mallocManaged((void **)&dAllCodewordsColors, sizeof(*dAllCodewordsColors) * roundedTotalCodewords);
  mallocManaged((void **)&dPossibleSolutions, sizeof(*dPossibleSolutions) * roundedTotalCodewords);
  mallocManaged((void **)&dPossibleSolutionsColors, sizeof(*dPossibleSolutionsColors) * roundedTotalCodewords);
  mallocManaged((void **)&dScores, sizeof(*dScores) * roundedTotalCodewords);
  mallocManaged((void **)&dRemainingIsPossibleSolution, sizeof(*dRemainingIsPossibleSolution) * roundedTotalCodewords);

  fdCount = (roundedTotalCodewords / 32) + 1;
  mallocManaged((void **)&dFullyDiscriminatingCodewords, sizeof(*dFullyDiscriminatingCodewords) * fdCount);
}

template <uint8_t p, uint8_t c, Algo a, bool l>
CUDAGPUInterface<p, c, a, l>::~CUDAGPUInterface() {
  auto freeManaged = [](auto devPtr) {
    cudaError_t err = cudaFree(devPtr);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to free managed memory (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  };

  freeManaged(dAllCodewords);
  freeManaged(dAllCodewordsColors);
  freeManaged(dPossibleSolutions);
  freeManaged(dPossibleSolutionsColors);
  freeManaged(dScores);
  freeManaged(dRemainingIsPossibleSolution);
  freeManaged(dFullyDiscriminatingCodewords);
}

template <uint8_t p, uint8_t c, Algo a, bool l>
bool CUDAGPUInterface<p, c, a, l>::gpuAvailable() const {
  return true;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
uint32_t *CUDAGPUInterface<p, c, a, l>::getAllCodewordsBuffer() {
  return dAllCodewords;
}
template <uint8_t p, uint8_t c, Algo a, bool l>
unsigned __int128 *CUDAGPUInterface<p, c, a, l>::getAllCodewordsColorsBuffer() {
  return dAllCodewordsColors;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
void CUDAGPUInterface<p, c, a, l>::setAllCodewordsCount(uint32_t count) {
  // TODO: this is redundant for this impl, and likely for the Metal impl too. Need to fix this up.
}

template <uint8_t p, uint8_t c, Algo a, bool l>
void CUDAGPUInterface<p, c, a, l>::syncAllCodewords(uint32_t count) {
  // TODO: let it page fault for now, come back and add movement hints if necessary.
}

template <uint8_t p, uint8_t c, Algo a, bool l>
uint32_t *CUDAGPUInterface<p, c, a, l>::getPossibleSolutionsBuffer() {
  return dPossibleSolutions;
}
template <uint8_t p, uint8_t c, Algo a, bool l>
unsigned __int128 *CUDAGPUInterface<p, c, a, l>::getPossibleSolutionsColorsBuffer() {
  return dPossibleSolutionsColors;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
void CUDAGPUInterface<p, c, a, l>::setPossibleSolutionsCount(uint32_t count) {
  possibleSolutionsCount = count;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
void CUDAGPUInterface<p, c, a, l>::sendComputeCommand() {
  cudaError_t err = cudaSuccess;
  err = cudaMemset(dFullyDiscriminatingCodewords, 0, sizeof(*dFullyDiscriminatingCodewords) * fdCount);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to clear FDC buffer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  subsettingAlgosKernel<p, a><<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      dAllCodewords, reinterpret_cast<const uint4 *>(dAllCodewordsColors), possibleSolutionsCount, dPossibleSolutions,
      reinterpret_cast<uint4 *>(dPossibleSolutionsColors), dScores, dRemainingIsPossibleSolution,
      dFullyDiscriminatingCodewords);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch subsettingAlgosKernel kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to sync device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template <uint8_t p, uint8_t c, Algo a, bool l>
uint32_t *CUDAGPUInterface<p, c, a, l>::getScores() {
  return dScores;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
bool *CUDAGPUInterface<p, c, a, l>::getRemainingIsPossibleSolution() {
  return dRemainingIsPossibleSolution;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
uint32_t *CUDAGPUInterface<p, c, a, l>::getFullyDiscriminatingCodewords(uint32_t &count) {
  count = fdCount;
  return dFullyDiscriminatingCodewords;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
std::string CUDAGPUInterface<p, c, a, l>::getGPUName() {
  // mmmfixme
  return "NEED GPU NAME";
}

// -----------------------------------------------------------------------------------
// Explicit specializations
//
// TODO: I ought to be able to get rid of these, but I need to try to wrangle the
//   conditional build stuff and compiler used for each file all the way up to main to allow
//   the templates to be included everywhere.

#define INST_PCL(p, c, l)                                       \
  template class CUDAGPUInterface<p, c, Algo::Knuth, l>;        \
  template class CUDAGPUInterface<p, c, Algo::MostParts, l>;    \
  template class CUDAGPUInterface<p, c, Algo::ExpectedSize, l>; \
  template class CUDAGPUInterface<p, c, Algo::Entropy, l>;

#define INST_CL(c, l) \
  INST_PCL(2, c, l)   \
  INST_PCL(3, c, l)   \
  INST_PCL(4, c, l)   \
  INST_PCL(5, c, l)   \
  INST_PCL(6, c, l)   \
  INST_PCL(7, c, l)   \
  INST_PCL(8, c, l)

#define INST_L(l) \
  INST_CL(2, l)   \
  INST_CL(3, l)   \
  INST_CL(4, l)   \
  INST_CL(5, l)   \
  INST_CL(6, l)   \
  INST_CL(7, l)   \
  INST_CL(8, l)   \
  INST_CL(9, l)   \
  INST_CL(10, l)  \
  INST_CL(11, l)  \
  INST_CL(12, l)  \
  INST_CL(13, l)  \
  INST_CL(14, l)  \
  INST_CL(15, l)

INST_L(true)
INST_L(false)
