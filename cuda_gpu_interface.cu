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

static __inline__ __host__ __device__ uchar4 make_uchar4(unsigned int i) { return *reinterpret_cast<uchar4 *>(&i); }

// https://godbolt.org/z/36735xfqa
template <uint pinCount>
__device__ uint scoreCodewords(const uint32_t &secret, const uint4 &secretColors, uint32_t &guess, uint4 &guessColors) {
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
  //  printf("%d %d\n", b, allHits - b);
  return b * pinCount - ((b - 1) * b) / 2 + allHits;
}

// The common portion of the kernels which scores all possible solutions against a given guess and computes subset
// sizes and whether or not the guess is a possible solution.
template <uint32_t pinCount>
__device__ bool computeSubsetSizes(int totalScores, uint32_t *scoreCounts, const uint32_t &secret,
                                   const uint4 &secretColors, /*constant*/ uint32_t &possibleSolutionsCount,
                                   uint32_t *possibleSolutions, uint4 *possibleSolutionsColors) {
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
// for whether or not the codeword is a possible solution.
//
// Finally, there's threadgroup memory for each thread with enough room for all of the intermediate subset sizes.
// This is important: the first version held this in threadlocal memory and occupancy was terrible.

extern __shared__ uint32_t tgScoreCounts[];

template <uint32_t pinCount, Algo algo, bool supportFamilyMac2>
__global__ void subsettingAlgosKernel(const uint32_t *allCodewords, const uint4 *allCodewordsColors,
                                      uint32_t possibleSolutionsCount, uint32_t *possibleSolutions,
                                      uint4 *possibleSolutionsColors, uint32_t *scores,
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
  cout << "CUDA!" << endl;

  const uint64_t totalCodewords = Codeword<p, c>::totalCodewords;
  threadsPerBlock = std::min(128lu, totalCodewords);
  // nb: round up!
  numBlocks = (totalCodewords + threadsPerBlock - 1) / threadsPerBlock;
  uint32_t roundedTotalCodewords = numBlocks * threadsPerBlock;

  // NB: matches the def in the compute kernel.
  const int totalScores = ((p * (p + 3)) / 2) + 1;
  sharedMemSize = sizeof(uint32_t) * totalScores * threadsPerBlock;

  cudaError_t err = cudaSuccess;

  // mmmfixme: error handling
  err = cudaMallocManaged((void **)&dAllCodewords, sizeof(*dAllCodewords) * roundedTotalCodewords);
  err = cudaMallocManaged((void **)&dAllCodewordsColors, sizeof(*dAllCodewordsColors) * roundedTotalCodewords);
  err = cudaMallocManaged((void **)&dPossibleSolutions, sizeof(*dPossibleSolutions) * roundedTotalCodewords);
  err =
      cudaMallocManaged((void **)&dPossibleSolutionsColors, sizeof(*dPossibleSolutionsColors) * roundedTotalCodewords);
  err = cudaMallocManaged((void **)&dScores, sizeof(*dScores) * roundedTotalCodewords);
  err = cudaMallocManaged((void **)&dRemainingIsPossibleSolution,
                          sizeof(*dRemainingIsPossibleSolution) * roundedTotalCodewords);

  fdCount = (roundedTotalCodewords / 32) + 1;
  err = cudaMallocManaged((void **)&dFullyDiscriminatingCodewords, sizeof(*dFullyDiscriminatingCodewords) * fdCount);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template <uint8_t p, uint8_t c, Algo a, bool l>
CUDAGPUInterface<p, c, a, l>::~CUDAGPUInterface() {
  // Free device global memory
  cudaError_t err = cudaSuccess;
  err = cudaFree(dAllCodewords);
  err = cudaFree(dAllCodewordsColors);
  err = cudaFree(dPossibleSolutions);
  err = cudaFree(dPossibleSolutionsColors);
  err = cudaFree(dScores);
  err = cudaFree(dRemainingIsPossibleSolution);
  err = cudaFree(dFullyDiscriminatingCodewords);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
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
  // mmmfixme: redundant
}

template <uint8_t p, uint8_t c, Algo a, bool l>
void CUDAGPUInterface<p, c, a, l>::syncAllCodewords(uint32_t count) {
  // mmmfixme: let it page fault for now, come back and add movement hints if necessary.
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
  cudaMemset(dFullyDiscriminatingCodewords, 0, sizeof(*dFullyDiscriminatingCodewords) * fdCount);
  subsettingAlgosKernel<p, a, false><<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      dAllCodewords, reinterpret_cast<const uint4 *>(dAllCodewordsColors), possibleSolutionsCount, dPossibleSolutions,
      reinterpret_cast<uint4 *>(dPossibleSolutionsColors), dScores, dRemainingIsPossibleSolution,
      dFullyDiscriminatingCodewords);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaDeviceSynchronize();
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
  //  return dFullyDiscriminatingCodewords;
//  count = 0;
//  return nullptr;
  count = fdCount;
  return dFullyDiscriminatingCodewords;
}

template <uint8_t p, uint8_t c, Algo a, bool l>
std::string CUDAGPUInterface<p, c, a, l>::getGPUName() {
  // mmmfixme
  return "NEED GPU NAME";
}

#define SPEC(p, c, l)                                           \
  template class CUDAGPUInterface<p, c, Algo::Knuth, l>;        \
  template class CUDAGPUInterface<p, c, Algo::MostParts, l>;    \
  template class CUDAGPUInterface<p, c, Algo::ExpectedSize, l>; \
  template class CUDAGPUInterface<p, c, Algo::Entropy, l>;

SPEC(4, 6, true)
SPEC(4, 6, false)
SPEC(5, 8, false)
SPEC(8, 4, false)
SPEC(8, 5, false)
SPEC(8, 6, false)
SPEC(8, 7, false)

// Starting search for secret 3632, initial guess is 1122 with 1,296 possibilities.
//
// Tried guess 1122 against secret 3632 => 10
// Removing inconsistent possibilities... 256 remain.
// Selecting best guess: 1344	score: 212 (GPU)
//
// Tried guess 1344 against secret 3632 => 01
// Removing inconsistent possibilities... 44 remain.
// Selecting best guess: 3526	score: 37 (GPU)
//
// Tried guess 3526 against secret 3632 => 12
// Removing inconsistent possibilities... 7 remain.
// Selecting best guess: 1462	score: 6 (GPU)
//
// Tried guess 1462 against secret 3632 => 11
// Removing inconsistent possibilities... 1 remain.
// Only remaining solution must be correct: 3632
//
// Tried guess 3632 against secret 3632 => 40
// Solution found after 5 moves.
//
// Playing all 4 pin 6 color games using algorithm 'Knuth' for every possible secret...
// Total codewords: 1,296
// Initial guess: 1122
// Average number of turns was 4.4761
// Maximum number of turns over all possible secrets was 5 with secret 1116
// Elapsed time 0.0836s, average search 0.0645ms
// Codeword comparisons: CPU = 3,237,885, GPU = 0, total = 3,237,885
//
//  1: 1  2: 6  3: 62  4: 533  5: 694
