// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#define CUB_STDERR
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/zip_function.h>

#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>
#include <cuda/barrier>
#include <cuda/functional>
#include <new>
#include <vector>

#include "algos.hpp"
#include "codeword.hpp"

// CUDA implementation for playing all games at once
//
// See solver_cuda.hpp for an overview.
//
// nb: scores here are not the classic combination of black hits and white hits. A score's ordinal is (b(p + 1) -
// ((b - 1)b) / 2) + w. See docs/Score_Ordinals.md for details. By using the score's ordinal we can have densely packed
// set of counters to form the subset counts as we go.

// Counter's we'll use on both the CPU and GPU
template <typename SolverConfig>
struct Counters {
  using S = SolverCUDA<SolverConfig>;
  constexpr static int SCORES = find_counter(S::counterDescs, "Scores");
  constexpr static int TINY_REGIONS = find_counter(S::counterDescs, "Tiny Regions");
  constexpr static int TINY_GAMES = find_counter(S::counterDescs, "Tiny Games");
  constexpr static int FDOPT_REGIONS = find_counter(S::counterDescs, "FDOpt Regions");
  constexpr static int FDOPT_GAMES = find_counter(S::counterDescs, "FDOpt Games");
  constexpr static int BIG_REGIONS = find_counter(S::counterDescs, "Big Regions");
  constexpr static int ACR_COUNT = find_counter(S::counterDescs, "ACr Count");
  constexpr static int ACR_SIZE = find_counter(S::counterDescs, "ACr Size");
};

// Mastermind scoring function
//
// This mirrors the scalar version very closely. It's the full counting method from Knuth, plus some fun bit twiddling
// hacks and SWAR action. This is O(1) using warp SIMD intrinsics.
//
// Find black hits with xor, which leaves zero nibbles on matches, then count the zeros in the result. This is a
// variation on determining if a word has a zero byte from https://graphics.stanford.edu/~seander/bithacks.html. This
// part ends with using the GPU's SIMD popcount() to count the zero nibbles.
//
// Next, color counts come pre-computed, and we can run over them and add up total hits, per Knuth[1], by aggregating
// min color counts between the secret and guess.
//
// Note this is specialized based on the number of colors in the game. Up to 8 colors are packed into an uint64_t and
// require fewer ops to reduce.
//
// Here's the asm for an early draft: https://godbolt.org/z/n1GE5P5GP The current code has a bunch of dependencies that
// make a quick compiler explorer link hard.
template <typename SolverConfig>
__device__ uint scoreCodewords(const uint32_t secret, const uint4 secretColors, const uint32_t guess,
                               const uint4 guessColors) {
  constexpr uint unusedPinsMask = 0xFFFFFFFFu & ~((1lu << SolverConfig::PIN_COUNT * 4u) - 1);
  uint v = secret ^ guess;  // Matched pins are now 0.
  v |= unusedPinsMask;      // Ensure that any unused pin positions are non-zero.
  uint r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint b = __popc(r);

  uint allHits;
  if constexpr (SolverConfig::CodewordT::isSize2()) {
    uint mins3 = __vminu4(secretColors.z, guessColors.z);
    uint mins4 = __vminu4(secretColors.w, guessColors.w);
    allHits = __vsadu4(mins3, 0);
    allHits += __vsadu4(mins4, 0);
  } else {
    static_assert(SolverConfig::CodewordT::isSize4());
    uint mins1 = __vminu4(secretColors.x, guessColors.x);
    uint mins2 = __vminu4(secretColors.y, guessColors.y);
    uint mins3 = __vminu4(secretColors.z, guessColors.z);
    uint mins4 = __vminu4(secretColors.w, guessColors.w);
    allHits = __vsadu4(mins1, 0);
    allHits += __vsadu4(mins2, 0);
    allHits += __vsadu4(mins3, 0);
    allHits += __vsadu4(mins4, 0);
  }

  return ((b * (2 * SolverConfig::PIN_COUNT + 1 - b)) / 2) + allHits;
}

// Score all possible solutions against a given secret and compute subset sizes, which are the number of codewords per
// score.
template <typename SolverConfig, typename SubsetSizeT, typename CodewordT>
__device__ void computeSubsetSizes(SubsetSizeT* __restrict__ subsetSizes, const uint32_t secret,
                                   const uint4 secretColors, const CodewordT* __restrict__ regionIDsAsCodeword,
                                   uint32_t regionStart, uint32_t regionLength) {
  for (int64_t i = regionStart; i < regionStart + regionLength; i++) {
    auto& ps = regionIDsAsCodeword[i];
    uint score = scoreCodewords<SolverConfig>(secret, secretColors, ps.packedCodeword(), ps.packedColorsCUDA());
    SolverConfig::ALGO::accumulateSubsetSize(subsetSizes[score]);
  }
}

// TODO: this is an attempt to stagger access to PS across all threads in the block, to try to parallelize and coalesce
// reads. It's strictly worse than the normal way above, though I expected it to do better. Too much overhead? Bad
// theory? Shrinking bs to 16 or 8 goes faster than blockDim.x
template <typename SolverConfig, typename SubsetSizeT, typename CodewordT>
__device__ void computeSubsetSizesStaggered(SubsetSizeT* __restrict__ subsetSizes, const uint32_t secret,
                                            const uint4 secretColors, const CodewordT* __restrict__ regionIDsAsCodeword,
                                            uint32_t regionStart, uint32_t regionLength) {
  auto s = regionStart;
  auto e = regionStart + regionLength;
  auto bs = blockDim.x;
  for (auto i = s; i < e; i += bs) {
    auto l = min(bs, e - i);
    auto j = i + (threadIdx.x % l);
    auto ke = i + l;
    for (auto k = i; k < ke; k++) {
      auto& ps = regionIDsAsCodeword[j++];
      if (j == ke) j = i;
      uint score = scoreCodewords<SolverConfig>(secret, secretColors, ps.packedCodeword(), ps.packedColorsCUDA());
      SolverConfig::ALGO::accumulateSubsetSize(subsetSizes[score]);
    }
  }
}

// Keeps an index into the all codewords vector together with a rank on the GPU, and whether this codeword is a
// possible solution.
struct IndexAndRank {
  uint32_t index;
  uint32_t rank;
  bool isPossibleSolution;
};

// Reducer for per-thread guesses, used for CUB per-block and device reductions.
struct IndexAndRankReducer {
  __device__ __forceinline__ IndexAndRank operator()(const IndexAndRank& a, const IndexAndRank& b) const {
    // Always take the best rank. If it's a tie, take the one that could be a solution. If that's a tie, take lexically
    // first.
    if (b.rank > a.rank) return b;
    if (b.rank < a.rank) return a;
    if (b.isPossibleSolution ^ a.isPossibleSolution) return b.isPossibleSolution ? b : a;
    return (b.index < a.index) ? b : a;
  }
};

// Holds all the constants we need to kick off the CUDA kernel for all the subsetting algos given a solver config.
// Computes how many threads per block, blocks needed, and importantly shared memory size. Can override the subset
// counter type to be smaller than the one given by the Strategy when we know the max subset size is small enough.
template <typename SolverConfig_, typename SubsetSizeOverrideT = uint32_t>
struct SubsettingAlgosKernelConfig {
  using SolverConfig = SolverConfig_;
  static constexpr uint8_t PIN_COUNT = SolverConfig::PIN_COUNT;
  static constexpr uint8_t COLOR_COUNT = SolverConfig::COLOR_COUNT;
  static constexpr bool LOG = SolverConfig::LOG;
  using ALGO = typename SolverConfig::ALGO;
  using CodewordT = typename SolverConfig::CodewordT;

  // Total scores = (PIN_COUNT * (PIN_COUNT + 3)) / 2, but +1 for imperfect packing.
  static constexpr int TOTAL_PACKED_SCORES = ((PIN_COUNT * (PIN_COUNT + 3)) / 2) + 1;

  using SubsetSizeT =
      typename std::conditional<sizeof(SubsetSizeOverrideT) < sizeof(typename SolverConfig::SubsetSizeT),
                                SubsetSizeOverrideT, typename SolverConfig::SubsetSizeT>::type;

  // This subset size is good given the PS size, or this is the default type provided by the Strategy.
  // No subset can be larger than PS, but a single subset may equal PS in the worst case.
  __host__ __device__ static bool shouldUseType(uint32_t possibleSolutionsCount) {
    return possibleSolutionsCount < cuda::std::numeric_limits<SubsetSizeT>::max() ||
           sizeof(SubsetSizeOverrideT) == sizeof(typename SolverConfig::SubsetSizeT);
  }

  // Max threads we could put in a group given how much shared memory space we need for packed subset counters.
  // This is rounded down to the prior power of two to satisfy the final reduction step.
  template <typename T>
  __host__ __device__ constexpr static uint32_t maxThreadsFromSubsetType() {
    uint32_t sharedMemSize = 48 * 1024;  // Default on 8.6
    uint32_t sharedMemPerThread = sizeof(T) * TOTAL_PACKED_SCORES;
    uint32_t threadsPerBlock = nextPowerOfTwo((sharedMemSize / sharedMemPerThread) / 2);
    return threadsPerBlock;
  }

  // How many threads will be put in each block. Always at least one warp, but no more than 512 (which needs to be tuned
  // more; 512 is picked based on results from 8p5c runs on MostParts and Knuth.)
  template <typename T>
  __host__ __device__ constexpr static uint32_t threadsPerBlock(const uint64_t totalCodewords) {
    return cudaExtra::std::clamp(
        cudaExtra::std::min(static_cast<uint64_t>(maxThreadsFromSubsetType<T>()), totalCodewords), 32ul, 512ul);
  }
  static constexpr uint32_t THREADS_PER_BLOCK = threadsPerBlock<SubsetSizeT>(CodewordT::TOTAL_CODEWORDS);

  // How many blocks we'll launch. This is rounded up to ensure we capture the last partial block. All kernels are
  // written to tolerate an incomplete final block.
  __host__ __device__ constexpr static uint32_t numBlocks(const uint64_t totalCodewords,
                                                          const uint32_t threadsPerBlock) {
    return (totalCodewords + threadsPerBlock - 1) / threadsPerBlock;
  }
  static constexpr uint32_t NUM_BLOCKS = numBlocks(CodewordT::TOTAL_CODEWORDS, THREADS_PER_BLOCK);

  // These are the worst-case values over all types this config will be specialized with. Currently, those are 1, 2, and
  // 4 byte types. We use the most blocks with the largest type, but we need the most space for codewords with the
  // smallest type since the block size is larger, and we round up a full block.
  static constexpr uint32_t LARGEST_NUM_BLOCKS =
      numBlocks(CodewordT::TOTAL_CODEWORDS, threadsPerBlock<uint32_t>(CodewordT::TOTAL_CODEWORDS));

  using BlockReduce = cub::BlockReduce<IndexAndRank, THREADS_PER_BLOCK>;

  union SharedMemLayout {
    SubsetSizeT subsetSizes[TOTAL_PACKED_SCORES * THREADS_PER_BLOCK];
    typename BlockReduce::TempStorage reducerTmpStorage;
    IndexAndRank aggregate;  // Ensure alignment for these
  };
};

// Little tests
using testConfig = SubsettingAlgosKernelConfig<SolverConfig<8, 5, false, Algos::Knuth>>;
static_assert(nextPowerOfTwo(uint32_t(136)) == 256);
static_assert(testConfig::maxThreadsFromSubsetType<uint32_t>() == 256);
static_assert(testConfig::numBlocks(testConfig::CodewordT::TOTAL_CODEWORDS,
                                    testConfig::threadsPerBlock<uint32_t>(testConfig::CodewordT::TOTAL_CODEWORDS)) ==
              1526);

// This takes two sets of codewords: the "all codewords" set, which is every possible codeword, and the "possible
// solutions" set. The all codewords set is placed into GPU memory once at program start and remains constant. The
// possible solutions set changes each time, both content and length, and is a sub-set of allCodewords.
//
// All codeword pairs are scored and subset sizes computed, then each codeword is ranked for the algorithm we're
// running. Finally, each block computes the best ranked codeword in the group, and we look for fully discriminating
// codewords.
//
// Output is an array of IndexAndRanks for the best selections from each block, and a single fully discriminating
// guess.
//
// Finally, there's shared block memory for each thread with enough room for all the intermediate subset sizes,
// reduction space, etc.
template <typename SubsettingAlgosKernelConfig, typename CodewordT>
__global__ void subsettingAlgosKernel(const CodewordT* __restrict__ allCodewords,
                                      const CodewordT* __restrict__ regionIDsAsCodeword,
                                      const uint32_t* __restrict__ regionIDsAsIndex, uint32_t regionStart,
                                      uint32_t regionLength, uint32_t** __restrict__ nextMovesVecs,
                                      uint32_t nextMovesVecsSize, IndexAndRank* __restrict__ perBlockSolutions,
                                      const uint32_t* __restrict__ acr, uint32_t acrLength) {
  __shared__ typename SubsettingAlgosKernelConfig::SharedMemLayout sharedMem;

  const uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  bool isPossibleSolution = false;
  uint32_t rank = 0;  // A rank of 0 will prevent used or invalid codewords from being chosen.

  // Initially assume we're working on the full AC
  auto totalCodewords = SubsettingAlgosKernelConfig::CodewordT::TOTAL_CODEWORDS;
  uint32_t acIndex = tidGrid;

  // If we do have a ACr, then reduce the number of codewords we're working on, and indirect through it to get the
  // actual codeword from AC.
  if (acr != nullptr) {
    totalCodewords = acrLength;
  }

  if (tidGrid < totalCodewords) {
    if (acr != nullptr) {
      acIndex = acr[tidGrid];
    }

    auto subsetSizes = &sharedMem.subsetSizes[threadIdx.x * SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES];
    for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

    computeSubsetSizes<SubsettingAlgosKernelConfig::SolverConfig>(subsetSizes, allCodewords[acIndex].packedCodeword(),
                                                                  allCodewords[acIndex].packedColorsCUDA(),
                                                                  regionIDsAsCodeword, regionStart, regionLength);

    auto possibleSolutionsCount = regionLength;
    isPossibleSolution = subsetSizes[SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES - 1] > 0;

    using ALGO = typename SubsettingAlgosKernelConfig::SolverConfig::ALGO;
    typename ALGO::RankingAccumulatorType rankingAccumulator{};
    for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) {
      if (subsetSizes[i] > 0) {
        ALGO::accumulateRanking(rankingAccumulator, subsetSizes[i], possibleSolutionsCount);
      }
    }

    rank = ALGO::computeRank(rankingAccumulator, possibleSolutionsCount);

    // Use the list of next moves sets to discard used codewords. nb: -1 to skip the new set.
    // TODO: I'd like to improve this. Ideally we wouldn't do this for low ranked guesses that won't be picked anyway.
    for (int i = 0; i < nextMovesVecsSize - 1; i++) {
      if (acIndex == nextMovesVecs[i][regionIDsAsIndex[regionStart]]) {
        rank = 0;
        break;
      }
    }
  }

  // Reduce to find the best solution we have in this block. This keeps the codeword index, rank, and possible solution
  // indicator together.
  __syncthreads();
  IndexAndRank iar{acIndex, rank, isPossibleSolution};
  IndexAndRank bestSolution =
      typename SubsettingAlgosKernelConfig::BlockReduce(sharedMem.reducerTmpStorage).Reduce(iar, IndexAndRankReducer());

  if (threadIdx.x == 0) {
    perBlockSolutions[blockIdx.x] = bestSolution;
  }
}

// Reduce the per-block best guesses from subsettingAlgosKernel to generate a single, best guess. This is then set
// as the next move for the region.
template <uint32_t blockSize>
__global__ void reduceBestGuess(IndexAndRank* __restrict__ perBlockSolutions, const uint32_t solutionsCount,
                                const uint32_t* __restrict__ regionIDsAsIndex, uint32_t* __restrict__ nextMoves,
                                const int regionStart, const int regionLength) {
  uint32_t idx = threadIdx.x;
  IndexAndRankReducer reduce;
  IndexAndRank bestGuess{0, 0, false};
  for (uint32_t i = idx; i < solutionsCount; i += blockSize) {
    bestGuess = reduce(bestGuess, perBlockSolutions[i]);
  }

  __shared__ IndexAndRank shared[blockSize];
  shared[idx] = bestGuess;
  __syncthreads();

  for (uint32_t size = blockSize / 2; size > 0; size /= 2) {
    if (idx < size) {
      shared[idx] = reduce(shared[idx], shared[idx + size]);
    }
    __syncthreads();
  }

  for (uint32_t i = idx; i < regionLength; i += blockSize) {
    nextMoves[regionIDsAsIndex[i + regionStart]] = shared[0].index;
  }
}

// Runs the full kernel for all the subsetting algorithms, plus the reducer afterwards. Results in nextMoves populated
// with the best guess for the entire region.
template <typename SubsettingAlgosKernelConfig, typename CodewordT>
__device__ void launchSubsettingKernel(const CodewordT* __restrict__ allCodewords,
                                       const CodewordT* __restrict__ regionIDsAsCodeword,
                                       const uint32_t* __restrict__ regionIDsAsIndex, uint32_t* __restrict__ nextMoves,
                                       const uint32_t regionStart, const uint32_t regionLength,
                                       uint32_t** __restrict__ nextMovesVecs, uint32_t nextMovesVecsSize,
                                       IndexAndRank* __restrict__ perBlockSolutions,
                                       unsigned long long int* __restrict__ deviceCounters,
                                       const uint32_t* __restrict__ acr, uint32_t acrLength) {
  // There's a constant def for blocks, threads, and shared mem layout based on |AC|. With the case equivalence opt,
  // those values are too large as |ACr| < |AC|, often by quite a lot. Adjust the number of blocks here. The threads per
  // block and shared mem size won't change, and they're too large, but that's not the end of the world. Ideally I'd
  // switch to dynamic shared memory, but I'm lazy.
  auto numBlocks = SubsettingAlgosKernelConfig::NUM_BLOCKS;
  if (acr != nullptr) {
    numBlocks = SubsettingAlgosKernelConfig::numBlocks(acrLength, SubsettingAlgosKernelConfig::THREADS_PER_BLOCK);
  }

  subsettingAlgosKernel<SubsettingAlgosKernelConfig><<<numBlocks, SubsettingAlgosKernelConfig::THREADS_PER_BLOCK>>>(
      allCodewords, regionIDsAsCodeword, regionIDsAsIndex, regionStart, regionLength, nextMovesVecs, nextMovesVecsSize,
      perBlockSolutions, acr, acrLength);
  CubDebug(cudaGetLastError());

  // nb: block size on this one must be a power of 2
  reduceBestGuess<128>
      <<<1, 128>>>(perBlockSolutions, numBlocks, regionIDsAsIndex, nextMoves, regionStart, regionLength);
  CubDebug(cudaGetLastError());

  atomicAdd(&deviceCounters[Counters<typename SubsettingAlgosKernelConfig::SolverConfig>::SCORES],
            static_cast<uint64_t>(acrLength) * regionLength);
}

// Holds all the constants we need to kick off the CUDA kernel for all the fully discriminating optimization given a
// solver config.
template <typename SolverConfig_>
struct FDOptKernelConfig {
  using SolverConfig = SolverConfig_;
  static constexpr uint8_t PIN_COUNT = SolverConfig::PIN_COUNT;
  static constexpr uint8_t COLOR_COUNT = SolverConfig::COLOR_COUNT;
  static constexpr bool LOG = SolverConfig::LOG;
  using CodewordT = typename SolverConfig::CodewordT;

  // Total scores = (PIN_COUNT * (PIN_COUNT + 3)) / 2, but +1 for imperfect packing.
  static constexpr int TOTAL_PACKED_SCORES = ((PIN_COUNT * (PIN_COUNT + 3)) / 2) + 1;

  using SubsetSizeT = uint8_t;

  static constexpr uint32_t THREADS_PER_BLOCK = 32;
  static constexpr uint32_t NUM_BLOCKS = 1;

  using SmallOptsBlockReduce = cub::WarpReduce<uint>;

  union SharedMemLayout {
    SubsetSizeT subsetSizes[TOTAL_PACKED_SCORES * THREADS_PER_BLOCK];
    typename SmallOptsBlockReduce::TempStorage smallOptsReducerTmpStorage;
  };
};

// Optimization from [2]: if the possible solution set is smaller than the number of possible scores, and if one
// codeword can fully discriminate all the possible solutions (i.e., it produces a different score for each one), then
// play it right away since it will tell us the winner.
//
// This compares PS with itself looking for a fully discriminating guess, and falls back to the full algo if none is
// found.
//
// This is an interesting shortcut. It doesn't change the results of the subsetting algorithms at all: average turns,
// max turns, max secret, and the full histograms all remain precisely the same. What does change is the number of
// scores computed, and the run time.
//
// nb: one block, one warp for this one. Max region length is 45 for 8 pin games, which is our pin max, so fewer than
// half the threads even have to loop to pickup all the data, and we get away with a single warp reduction.
template <typename FDOptKernelConfig, typename SubsettingAlgosKernelConfig, typename CodewordT>
__global__ void fullyDiscriminatingOpt(const CodewordT* __restrict__ allCodewords,
                                       const CodewordT* __restrict__ regionIDsAsCodeword,
                                       const uint32_t* __restrict__ regionIDsAsIndex, uint32_t regionStart,
                                       uint32_t regionLength, uint32_t* __restrict__ nextMoves,
                                       uint32_t** __restrict__ nextMovesVecs, uint32_t nextMovesVecsSize,
                                       IndexAndRank* __restrict__ perBlockSolutions,
                                       unsigned long long int* __restrict__ deviceCounters,
                                       const uint32_t* __restrict__ acr, uint32_t acrLength) {
  assert(blockIdx.x == 0);   // Single block
  assert(blockDim.x == 32);  // Single warp

  using SolverConfig = typename FDOptKernelConfig::SolverConfig;
  __shared__ typename FDOptKernelConfig::SharedMemLayout sharedMem;
  uint result = cuda::std::numeric_limits<uint>::max();

  for (uint idx = threadIdx.x; idx < regionLength; idx += blockDim.x) {
    auto subsetSizes = &sharedMem.subsetSizes[idx * SolverConfig::TOTAL_PACKED_SCORES];
    for (int i = 0; i < SolverConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

    computeSubsetSizes<SolverConfig>(subsetSizes, regionIDsAsCodeword[idx + regionStart].packedCodeword(),
                                     regionIDsAsCodeword[idx + regionStart].packedColorsCUDA(), regionIDsAsCodeword,
                                     regionStart, regionLength);

    uint32_t totalUsedSubsets = 0;
    for (int i = 0; i < SolverConfig::TOTAL_PACKED_SCORES; i++) {
      if (subsetSizes[i] > 0) {
        totalUsedSubsets++;
      }
    }

    if (totalUsedSubsets == regionLength) {
      result = min(result, regionIDsAsIndex[idx + regionStart]);
    }
  }

  __syncthreads();
  uint bestSolution =
      typename FDOptKernelConfig::SmallOptsBlockReduce(sharedMem.smallOptsReducerTmpStorage).Reduce(result, cub::Min());

  if (threadIdx.x == 0) {
    atomicAdd(&deviceCounters[Counters<SolverConfig>::SCORES], regionLength * regionLength);
    if (bestSolution < cuda::std::numeric_limits<uint>::max()) {
      for (int i = 0; i < regionLength; i++) {
        nextMoves[regionIDsAsIndex[i + regionStart]] = bestSolution;
      }
    } else {
      // Fallback on the big kernel
      launchSubsettingKernel<SubsettingAlgosKernelConfig>(
          allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart, regionLength, nextMovesVecs,
          nextMovesVecsSize, perBlockSolutions, deviceCounters, acr, acrLength);
    }
  }
}

// Find the next guess for a group of regions. Each thread figures out the best kernel to launch for a region.
// The optimization for small regions which could have a fully discriminating guess is handled here for now as well.
template <typename SolverConfig, typename CodewordT>
__global__ void nextGuessForRegions(const CodewordT* __restrict__ allCodewords,
                                    const CodewordT* __restrict__ regionIDsAsCodeword,
                                    const uint32_t* __restrict__ regionIDsAsIndex, uint32_t* __restrict__ nextMoves,
                                    const uint32_t* __restrict__ regionStarts,
                                    const uint32_t* __restrict__ regionLengths, const uint32_t offset,
                                    const uint32_t regionCount, uint32_t** __restrict__ nextMovesVecs,
                                    uint32_t nextMovesVecsSize, IndexAndRank* __restrict__ perBlockSolutionsPool,
                                    unsigned long long int* __restrict__ deviceCounters,
                                    const uint32_t* __restrict__ acrBuffer, const uint32_t* __restrict__ acrStarts,
                                    const uint32_t* __restrict__ acrLengths) {
  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidGrid < regionCount) {
    auto regionStart = regionStarts[offset + tidGrid];
    auto regionLength = regionLengths[offset + tidGrid];
    auto perBlockSolutions =
        &perBlockSolutionsPool[SubsettingAlgosKernelConfig<SolverConfig, uint32_t>::LARGEST_NUM_BLOCKS * tidGrid];

    // If we have ACr data then resolve the location of ACr in the buffer and its length.
    const uint32_t* acr = nullptr;
    uint32_t acrLength = CodewordT::TOTAL_CODEWORDS;
    if (acrStarts != nullptr) {
      auto acrStart = acrStarts[offset + tidGrid];
      if (acrStart != cuda::std::numeric_limits<uint32_t>::max()) {
        acr = &acrBuffer[acrStart];
        acrLength = acrLengths[offset + tidGrid];
      }
    }

    using config8 = SubsettingAlgosKernelConfig<SolverConfig, uint8_t>;
    using config16 = SubsettingAlgosKernelConfig<SolverConfig, uint16_t>;
    using config32 = SubsettingAlgosKernelConfig<SolverConfig, uint32_t>;

    if (config8::shouldUseType(regionLength)) {
      if (regionLength < SolverConfig::TOTAL_PACKED_SCORES) {
        using configFDOpt = FDOptKernelConfig<SolverConfig>;
        fullyDiscriminatingOpt<configFDOpt, config8><<<1, configFDOpt::THREADS_PER_BLOCK>>>(
            allCodewords, regionIDsAsCodeword, regionIDsAsIndex, regionStart, regionLength, nextMoves, nextMovesVecs,
            nextMovesVecsSize, perBlockSolutions, deviceCounters, acr, acrLength);
      } else {
        launchSubsettingKernel<config8>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart,
                                        regionLength, nextMovesVecs, nextMovesVecsSize, perBlockSolutions,
                                        deviceCounters, acr, acrLength);
      }
    } else if (config16::shouldUseType(regionLength)) {
      launchSubsettingKernel<config16>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart,
                                       regionLength, nextMovesVecs, nextMovesVecsSize, perBlockSolutions,
                                       deviceCounters, acr, acrLength);
    } else {
      launchSubsettingKernel<config32>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart,
                                       regionLength, nextMovesVecs, nextMovesVecsSize, perBlockSolutions,
                                       deviceCounters, acr, acrLength);
    }
  }
}

// Handle all the 1 & 2 size regions by selecting the first guess.
__global__ void nextGuessTiny(const uint32_t* __restrict__ regionIDsAsIndex, uint32_t* __restrict__ nextMoves,
                              const uint32_t* __restrict__ regionStarts, const uint32_t* __restrict__ regionLengths,
                              const uint32_t regionCount) {
  for (uint32_t runIndex = threadIdx.x; runIndex < regionCount; runIndex += blockDim.x) {
    auto regionStart = regionStarts[runIndex];
    auto regionLength = regionLengths[runIndex];
    if (regionLength == 1) {
      nextMoves[regionIDsAsIndex[regionStart]] = regionIDsAsIndex[regionStart];
    } else if (regionLength == 2) {
      auto lexicallyFirst = min(regionIDsAsIndex[regionStart], regionIDsAsIndex[regionStart + 1]);
      nextMoves[regionIDsAsIndex[regionStart]] = lexicallyFirst;
      nextMoves[regionIDsAsIndex[regionStart + 1]] = lexicallyFirst;
    }
  }
}

// See the overview of the algorithm in solver_cuda.hpp.
//
// This builds all the buffers we need on-device for gameplay state, then loops playing all games a turn at a time
// until all games have been won. I've used Thrust to try to keep the bulk of it fairly simple.
//
// Note: this uses a lot of device memory right now. Need to figure out the max size game as is and go from there.
//  - the pins of a packed codeword are 32bits. Could drop the colors and re-compute them as needed on-device.
//  - could make a packed regionID w/ 6bit scores
//  - region starts and lengths could be delta coded and variable size, etc.
//
// Much of this work is a serial list of Thrust kernels and could be a parallel graph, but the time spent outside the
// main subsetting kernel is a tiny fraction of the overall work right now, so keeping it simple.
template <typename SolverConfig>
std::chrono::nanoseconds SolverCUDA<SolverConfig>::playAllGames(uint32_t packedInitialGuess) {
  constexpr static bool LOG = SolverConfig::LOG;

  auto overallStartTime = chrono::high_resolution_clock::now();

  // All codewords go to the device once
  thrust::device_vector<CodewordT> dAllCodewords = CodewordT::getAllCodewords();

  // Hold the next moves in a parallel vector to the all codewords vector. We need one such vector per turn played,
  // so we can have a "used codewords" list, and so we can form the full output graphs of moves when done.
  // TODO: This is a lot of memory. Could stream these out to the host while doing other work to have just one or two
  //  on-device at a time, but that doesn't support the used codewords set. Feel like there's an alternative here.
  constexpr static int MAX_SUPPORTED_TURNS = 16;
  // Use device vectors for the storage, and keep em in a host vector to get them all freed at the end.
  thrust::host_vector<uint32_t*> hNextMovesDeviceVecs(MAX_SUPPORTED_TURNS);
  thrust::host_vector<thrust::device_vector<uint32_t>> hNextMovesStorage(hNextMovesDeviceVecs.size());
  for (int i = 0; i < MAX_SUPPORTED_TURNS; i++) {
    hNextMovesStorage[i] = thrust::device_vector<uint32_t>(dAllCodewords.size());
    hNextMovesDeviceVecs[i] = thrust::raw_pointer_cast(hNextMovesStorage[i].data());
  }
  thrust::device_vector<uint32_t*> dNextMovesVecs(hNextMovesDeviceVecs.size());
  thrust::copy(hNextMovesDeviceVecs.begin(), hNextMovesDeviceVecs.end(), dNextMovesVecs.begin());
  uint32_t** pdNextMovesVecs = thrust::raw_pointer_cast(dNextMovesVecs.data());

  // Starting case: all games playable, same initial guess.
  int nextMovesVecsSize = 0;
  auto dNextMoves = thrust::device_pointer_cast(hNextMovesDeviceVecs[nextMovesVecsSize++]);
  thrust::fill(dNextMoves, dNextMoves + dAllCodewords.size(), CodewordT::computeOrdinal(packedInitialGuess));
  auto pdNextMoves = thrust::raw_pointer_cast(dNextMoves);

  // All region ids start empty, with their index set to the sequence of all codewords
  thrust::host_vector<RegionID> hRegionIDs(dAllCodewords.size());
  for (uint32_t i = 0; i < hRegionIDs.size(); i++) hRegionIDs[i].index = i;
  thrust::device_vector<RegionID> dRegionIDs = hRegionIDs;

  // Space for the region locations
  thrust::device_vector<uint32_t> dRegionStarts(dRegionIDs.size());
  thrust::device_vector<uint32_t> dRegionLengths(dRegionIDs.size());
  thrust::host_vector<uint32_t> hRegionLengths(dRegionIDs.size());

  // Space for the intermediate reduction results out of the main subsetting algos kernel. We need a chunk of space for
  // every concurrent kernel execution, and we more or less blocks depending on the subset sizes. So allocate the max
  // number of blocks possible, one set per concurrent kernel.
  constexpr static size_t concurrentSubsettingKernels = 256;
  thrust::device_vector<IndexAndRank> dPerBlockSolutions(
      SubsettingAlgosKernelConfig<SolverConfig, uint32_t>::LARGEST_NUM_BLOCKS * concurrentSubsettingKernels);

  // Space to pre-process regions to codeword indices, or actual codewords. Helps speed up some later kernels as they
  // can avoid multiple reads due to indirection. At the expense of a decent amount of memory, though.
  thrust::device_vector<uint32_t> dRegionIDsAsIndex(dRegionIDs.size());
  thrust::device_vector<CodewordT> dRegionIDsAsCodeword(dRegionIDs.size());

  // Space for the Case Equivalence opts
  thrust::device_vector<uint32_t> dACrBuffer;
  thrust::device_vector<uint32_t> dACrStarts;
  thrust::device_vector<uint32_t> dACrLengths;
  thrust::device_vector<ZFColors> dZFColors(dRegionIDs.size());

  // A little space for some counters
  thrust::host_vector<unsigned long long int> hDeviceCounters(counters.size(), 0);
  thrust::device_vector<unsigned long long int> dDeviceCounters = hDeviceCounters;
  unsigned long long int* pdDeviceCounters = thrust::raw_pointer_cast(dDeviceCounters.data());

  int depth = 0;
  auto dRegionIDsEnd = dRegionIDs.end();  // The set of active games contracts as we go

  while (true) {
    auto startTime = chrono::high_resolution_clock::now();
    depth++;

    if (LOG) printf("\nDepth = %d\n", depth);

    // Score all games against their next guess, if any, which was given per-region. Append the score to the game's
    // region id.
    auto pdAllCodewords = thrust::raw_pointer_cast(dAllCodewords.data());
    thrust::for_each(
        dRegionIDs.begin(), dRegionIDsEnd, [depth, pdAllCodewords, pdNextMoves] __device__(RegionID & regionID) {
          auto cwi = regionID.index;
          uint8_t s = scoreCodewords<SolverConfig>(
              pdAllCodewords[cwi].packedCodeword(), pdAllCodewords[cwi].packedColorsCUDA(),
              pdAllCodewords[pdNextMoves[cwi]].packedCodeword(), pdAllCodewords[pdNextMoves[cwi]].packedColorsCUDA());
          regionID.append(s, depth);
        });
    counters[Counters<SolverConfig>::SCORES] += dRegionIDsEnd - dRegionIDs.begin();

    // Push won games to the end and focus on the remaining games
    dRegionIDsEnd = thrust::partition(dRegionIDs.begin(), dRegionIDsEnd,
                                      [] __device__(const RegionID& r) { return !r.isGameOver(); });

    if (LOG) cout << "Number of games left: " << commaString(dRegionIDsEnd - dRegionIDs.begin()) << endl;

    // If no games need new moves, then we're done
    if (dRegionIDsEnd - dRegionIDs.begin() == 0) break;

    // Sort all games by region id. Doesn't need to be stable since all of our reducers have to use the index to get
    // back lexical ordering anyway.
    thrust::sort(dRegionIDs.begin(), dRegionIDsEnd,
                 [] __device__(const RegionID& a, const RegionID& b) { return a.value < b.value; });

    // Get run length for each region. nb: discarding the keys since the computed starts are sufficient.
    auto regionCount =
        thrust::reduce_by_key(dRegionIDs.begin(), dRegionIDsEnd, thrust::constant_iterator<int>(1),
                              thrust::make_discard_iterator(), dRegionLengths.begin(),
                              [] __device__(const RegionID& a, const RegionID& b) { return a.value == b.value; })
            .second -
        dRegionLengths.begin();
    if (LOG) cout << "Number of regions: " << commaString(regionCount) << endl;

    // Now build starts for each region
    thrust::exclusive_scan(dRegionLengths.begin(), dRegionLengths.begin() + regionCount, dRegionStarts.begin());

    // Optimization for Symmetry and Case Equivalence
    //
    // Adapted from Ville[2], section 5.4. See docs/Symmetry_and_Case_Equivalence.ipynb for full details.
    // The first step is to gather the Zero and Free info for each region.
    if constexpr (SolverConfig::SYMOPT) {
      buildZerosAndFrees(pdAllCodewords, dRegionIDs, dRegionIDsEnd, regionCount, dRegionStarts, pdNextMovesVecs,
                         nextMovesVecsSize, dZFColors);

      // Sort the regions by length. Lets us batch up work for regions of different interesting sizes below. Include the
      // zero and free data, too.
      thrust::sort_by_key(dRegionLengths.begin(), dRegionLengths.begin() + regionCount,
                          thrust::make_zip_iterator(dRegionStarts.begin(), dZFColors.begin()));
    } else {
      // Sort the regions by length. Lets us batch up work for regions of different interesting sizes below
      thrust::sort_by_key(dRegionLengths.begin(), dRegionLengths.begin() + regionCount, dRegionStarts.begin());
    }

    // Have to take the hit and pull the region lengths back, so we can launch different kernels
    hRegionLengths = dRegionLengths;

    // How many regions are "tiny"? We'll process these separately below, and avoid doing any more CE work for them too.
    uint32_t tinyRegionCount = 0;
    uint32_t tinyGameCount = 0;
    for (uint32_t i = 0; i < regionCount && hRegionLengths[i] <= 2; i++) {
      tinyGameCount += hRegionLengths[tinyRegionCount];
      tinyRegionCount++;
    }

    // The next phase in the Case Equivalence opt is to gather the unique combinations of Zero and Free, and generate
    // all ACr from them. I'm building these all together for now, and assuming they will fit. When that fails to be
    // the case we'll have to generate as many as will fit in memory, launch the kernels we can, then loop.
    if constexpr (SolverConfig::SYMOPT) {
      buildAllACr(dZFColors, dAllCodewords, dRegionStarts, dRegionLengths, regionCount, tinyRegionCount, dACrBuffer,
                  dACrStarts, dACrLengths);
    }

    auto pdACrIBuffer = thrust::raw_pointer_cast(dACrBuffer.data());
    auto pdACrStarts = thrust::raw_pointer_cast(dACrStarts.data());
    auto pdACrLengths = thrust::raw_pointer_cast(dACrLengths.data());

    // TODO: these could probably be one zipped transform
    // TODO: re-test these. Trades a lot of device space for a small time gain, worth it?
    thrust::transform(dRegionIDs.begin(), dRegionIDsEnd, dRegionIDsAsIndex.begin(),
                      [] __device__(const RegionID& r) { return r.index; });
    thrust::transform(dRegionIDs.begin(), dRegionIDsEnd, dRegionIDsAsCodeword.begin(),
                      [pdAllCodewords] __device__(const RegionID& r) { return pdAllCodewords[r.index]; });

    if (LOG) {
      auto endTime = chrono::high_resolution_clock::now();
      chrono::duration<float> elapsedS = endTime - startTime;
      cout << "Phase 1 elapsed time " << commaString(elapsedS.count()) << "s" << endl;
      startTime = chrono::high_resolution_clock::now();
    }

    // For reach region:
    //   games with a win at the end of their region id get no new guess
    //   otherwise, find next guess using the region itself as the possible solutions set PS
    //
    // Also treat regions of different lengths specially. There are some simple opts for size 1 & 2 regions,
    // and a nice early shortcut for regions which can potentially be fully discriminated.
    auto pdRegionStarts = thrust::raw_pointer_cast(dRegionStarts.data());
    auto pdRegionLengths = thrust::raw_pointer_cast(dRegionLengths.data());
    auto pdRegionsAsIndex = thrust::raw_pointer_cast(dRegionIDsAsIndex.data());
    auto pdRegionsAsCodeword = thrust::raw_pointer_cast(dRegionIDsAsCodeword.data());

    // Advance to a fresh next moves vector
    pdNextMoves = thrust::raw_pointer_cast(hNextMovesDeviceVecs[nextMovesVecsSize++]);

    // Process "tiny" regions specially. They need virtually no work, but it all has to act on device memory, so we use
    // a single small kernel to take care of them all very, very quickly.
    if (tinyRegionCount > 0) {
      nextGuessTiny<<<1, 128>>>(pdRegionsAsIndex, pdNextMoves, pdRegionStarts, pdRegionLengths, tinyRegionCount);
    }

    // Kickoff the full subsetting kernel for each large region, with each kernel processing a chunk of regions at a
    // time. This is where all the time is spent.
    for (size_t offset = tinyRegionCount; offset < regionCount; offset += concurrentSubsettingKernels) {
      auto regionsToDo = min(concurrentSubsettingKernels, regionCount - offset);
      int threadsPerBlock = 4;  // Reduce dynamic launch parallelism by 4

      auto pdPerBlockSolutions = thrust::raw_pointer_cast(dPerBlockSolutions.data());
      nextGuessForRegions<SolverConfig><<<(regionsToDo + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
          pdAllCodewords, pdRegionsAsCodeword, pdRegionsAsIndex, pdNextMoves, pdRegionStarts, pdRegionLengths, offset,
          regionsToDo, pdNextMovesVecs, nextMovesVecsSize, pdPerBlockSolutions, pdDeviceCounters, pdACrIBuffer,
          pdACrStarts, pdACrLengths);
    }

    // Small regions are amenable to the fully discriminating opt. These are counted here, but really handled by
    // the normal kernel. I've processed these separately in other versions, but it makes minimal difference. All the
    // time is really spent on the big ones.
    uint32_t fdRegionCount = 0;
    uint32_t fdGameCount = 0;
    for (uint32_t i = tinyRegionCount; i < regionCount; i++) {
      if (hRegionLengths[i] < SolverConfig::TOTAL_PACKED_SCORES) {
        fdGameCount += hRegionLengths[i];
        fdRegionCount++;
      }
    }

    counters[Counters<SolverConfig>::TINY_REGIONS] += tinyRegionCount;
    counters[Counters<SolverConfig>::TINY_GAMES] += tinyGameCount;
    counters[Counters<SolverConfig>::FDOPT_REGIONS] += fdRegionCount;
    counters[Counters<SolverConfig>::FDOPT_GAMES] += fdGameCount;

    uint32_t bigRegionCount = regionCount - tinyRegionCount - fdRegionCount;
    counters[Counters<SolverConfig>::BIG_REGIONS] += bigRegionCount;

    if (LOG) {
      printf("Tiny regions: %s, totalling %s games\n", commaString(tinyRegionCount).c_str(),
             commaString(tinyGameCount).c_str());
      printf("Possibly fully discriminating regions: %s, totalling %s games\n", commaString(fdRegionCount).c_str(),
             commaString(fdGameCount).c_str());
      printf("Big regions: %s\n", commaString(bigRegionCount).c_str());
    }

    CubDebug(cudaDeviceSynchronize());

    if (LOG) {
      auto endTime = chrono::high_resolution_clock::now();
      chrono::duration<float> elapsedS = endTime - startTime;
      cout << "Phase 2 elapsed time " << commaString(elapsedS.count()) << "s" << endl;
    }

    if (depth == MAX_SUPPORTED_TURNS) {
      printf("\nMax depth reached, impl is broken!\n");
      break;
    }
  }

  auto overallEndTime = chrono::high_resolution_clock::now();

  if (LOG) cout << "Last actual depth: " << depth << endl;

  hDeviceCounters = dDeviceCounters;
  for (int i = 0; i < counterDescs.descs.size(); i++) {
    counters[i] += hDeviceCounters[i];
  }

  // Post-process for stats
  hRegionIDs = dRegionIDs;
  for (int i = 0; i < hRegionIDs.size(); i++) {
    auto c = hRegionIDs[i].countMovesPacked();
    this->maxDepth = max<size_t>(this->maxDepth, c);
    this->totalTurns += c;
  }

  // Copy solution data off the GPU, so we can use it to dump strategy graphs and other stats
  regionIDs = std::vector<RegionID>(hRegionIDs.begin(), hRegionIDs.end());

  for (int i = 0; i < hNextMovesStorage.size(); i++) {
    auto dNM = hNextMovesStorage[i];
    auto nm = vector<uint32_t>(dNM.size());
    thrust::copy(dNM.begin(), dNM.end(), nm.begin());
    nextMovesList.push_back(nm);
  }

  return overallEndTime - overallStartTime;
}

template <typename SolverConfig>
void SolverCUDA<SolverConfig>::dump() {
  Solver::dump<SolverConfig, CodewordT>(regionIDs);
}

template <typename SolverConfig>
vector<uint32_t> SolverCUDA<SolverConfig>::getGuessesForGame(uint32_t packedCodeword) {
  return Solver::getGuessesForGame<SolverCUDA, SolverConfig, CodewordT>(packedCodeword, regionIDs);
}

// Optimization for Symmetry and Case Equivalence
//
// Adapted from Ville[2], section 5.4. See docs/Symmetry_and_Case_Equivalence.ipynb for full details.

// Zero and Free sets for each region.
template <typename SolverConfig>
void SolverCUDA<SolverConfig>::buildZerosAndFrees(const CodewordT* pdAllCodewords,
                                                  thrust::device_vector<RegionID>& dRegionIDs,
                                                  thrust::device_vector<RegionID>::iterator& dRegionIDsEnd,
                                                  uint32_t regionCount, thrust::device_vector<uint32_t>& dRegionStarts,
                                                  uint32_t** pdNextMovesVecs, uint32_t nextMovesVecsSize,
                                                  thrust::device_vector<ZFColors>& dZFColors) {
  thrust::device_vector<uint32_t> dZFZero(dRegionIDs.size());
  thrust::device_vector<uint32_t> dZFFree(dRegionIDs.size());

  // Build Zero for each region, ordered with the current regions. This is a reduction of each region (PS).
  auto usedColorsToZerosMask = [] __device__(typename CodewordT::CT usedColors) -> uint32_t {
    uint32_t isZero = 0;
    for (uint8_t color = 1; color <= SolverConfig::COLOR_COUNT; color++) {
      if ((usedColors & 0xFF) == 0) {
        isZero |= (1 << color);
      }
      usedColors >>= 8;
    }
    if (__popc(isZero) == 1) return 0;
    return isZero;
  };

  auto zerosForRegions = thrust::reduce_by_key(
      dRegionIDs.begin(), dRegionIDsEnd,
      thrust::make_transform_iterator(
          dRegionIDs.begin(),
          [pdAllCodewords] __host__ __device__(const RegionID& v) { return pdAllCodewords[v.index].packedColors(); }),
      thrust::make_discard_iterator(), thrust::make_transform_output_iterator(dZFZero.begin(), usedColorsToZerosMask),
      [] __device__(const RegionID& a, const RegionID& b) { return a.value == b.value; },
      [] __device__(const typename CodewordT::CT a, const typename CodewordT::CT b) { return a | b; });
  assert(zerosForRegions.second.base() - dZFZero.begin() == regionCount);

  // Build Free for each region, ordered with the current regions. This is a loop over the used set for each region.
  auto pdRegionIDs = thrust::raw_pointer_cast(dRegionIDs.data());
  thrust::transform(dRegionStarts.begin(), dRegionStarts.begin() + regionCount, dZFFree.begin(),
                    [pdAllCodewords, pdNextMovesVecs, nextMovesVecsSize, pdRegionIDs] __device__(uint32_t regionStart) {
                      auto cwi = pdRegionIDs[regionStart].index;
                      typename CodewordT::CT playedColors = 0;
                      for (int i = 0; i < nextMovesVecsSize; i++) {
                        playedColors |= pdAllCodewords[pdNextMovesVecs[i][cwi]].packedColors();
                      }
                      uint32_t isFree = 0;
                      for (uint8_t color = 1; color <= SolverConfig::COLOR_COUNT; color++) {
                        if ((playedColors & 0xFF) == 0) {
                          isFree |= (1 << color);
                        }
                        playedColors >>= 8;
                      }
                      if (__popc(isFree) == 1) isFree = 0;
                      return isFree;
                    });

  // Build combined Zero/Free colors for each region, still in order of current regions & starts
  thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(dZFZero.begin(), dZFFree.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(dZFZero.begin() + regionCount, dZFFree.begin() + regionCount)),
      dZFColors.begin(), thrust::make_zip_function([] __device__(uint32_t z, uint32_t f) {
        // Zeros can't also be frees
        f &= ~z;
        return ZFColors{z, f};
      }));
}

// Build all ACr and leave their locations in device memory, associated with each region.
template <typename SolverConfig>
void SolverCUDA<SolverConfig>::buildAllACr(thrust::device_vector<ZFColors>& dZFColors,
                                           thrust::device_vector<CodewordT>& dAllCodewords,
                                           thrust::device_vector<uint32_t>& dRegionStarts,
                                           thrust::device_vector<uint32_t>& dRegionLengths, uint32_t regionCount,
                                           uint32_t tinyRegionCount, thrust::device_vector<uint32_t>& dACrBuffer,
                                           thrust::device_vector<uint32_t>& dACrStarts,
                                           thrust::device_vector<uint32_t>& dACrLengths) {
  constexpr static bool LOG = SolverConfig::LOG;

  // nb: skipping the tiny regions, and keeping region starts and lengths associated with the Zero and Free data.
  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(dRegionStarts.begin() + tinyRegionCount, dRegionLengths.begin() + tinyRegionCount));
  thrust::sort_by_key(dZFColors.begin() + tinyRegionCount, dZFColors.begin() + regionCount, begin);

  thrust::device_vector<ZFColors> dZFColorsUnique(regionCount);
  thrust::device_vector<uint32_t> dZFColorsUniqueLens(regionCount);
  auto dZFColorsUniqueEnd = thrust::reduce_by_key(dZFColors.begin() + tinyRegionCount, dZFColors.begin() + regionCount,
                                                  dRegionLengths.begin() + tinyRegionCount, dZFColorsUnique.begin(),
                                                  dZFColorsUniqueLens.begin());
  dZFColorsUnique.resize(thrust::distance(dZFColorsUnique.begin(), dZFColorsUniqueEnd.first));
  dZFColorsUniqueLens.resize(thrust::distance(dZFColorsUniqueLens.begin(), dZFColorsUniqueEnd.second));
  if (LOG) cout << "Unique combinations of Zero and Free colors: " << commaString(dZFColorsUnique.size()) << endl;

  // How much space will we need for all ACr?
  thrust::host_vector<ZFColors> hZFColorsUnique(dZFColorsUnique);
  thrust::host_vector<uint32_t> hZFColorsUniqueLens(dZFColorsUniqueLens);
  auto& acrCache = getACrCache();
  uint32_t totalACrEntries = 0;
  for (auto& k : hZFColorsUnique) {
    int zeroSize = popcount(k.zero);
    int freeSize = popcount(k.free);
    if (zeroSize == 0 && freeSize == 0) {
      continue;
    }
    int ck = SolverConfig::PIN_COUNT * 1000000 + SolverConfig::COLOR_COUNT * 10000 + zeroSize * 100 + freeSize;
    if (acrCache.contains(ck)) {
      totalACrEntries += acrCache[ck];
    } else {
      cout << "WARNING: acrCache[" << ck << "]: missing entry" << endl;
    }
  }
  if (LOG) cout << "Total ACrEntries: " << commaString(totalACrEntries) << endl;
  counters[Counters<SolverConfig>::ACR_SIZE] += totalACrEntries;
  dACrBuffer.resize(totalACrEntries);

  // Build every ACr, and a map of color combos to ACr position and length in the buffer.
  struct ACrLocation {
    uint32_t start;
    uint32_t len;
  };

  map<ZFColors, ACrLocation> zfColorsToACrLocations;
  uint32_t currentACrStart = 0;
  for (int i = 0; i < hZFColorsUnique.size(); i++) {
    auto& k = hZFColorsUnique[i];
    int zeroSize = popcount(k.zero);
    int freeSize = popcount(k.free);

    if (zeroSize == 0 && freeSize == 0) {
      zfColorsToACrLocations[k] = {cuda::std::numeric_limits<uint32_t>::max(),
                                   cuda::std::numeric_limits<uint32_t>::max()};
      continue;
    }

    int ck = SolverConfig::PIN_COUNT * 1000000 + SolverConfig::COLOR_COUNT * 10000 + zeroSize * 100 + freeSize;
    if (!acrCache.contains(ck)) {
      zfColorsToACrLocations[k] = {cuda::std::numeric_limits<uint32_t>::max(),
                                   cuda::std::numeric_limits<uint32_t>::max()};
      continue;
    }

    auto acrEnd = thrust::copy_if(
        thrust::make_counting_iterator((uint32_t)0), thrust::make_counting_iterator((uint32_t)dAllCodewords.size()),
        dAllCodewords.begin(), dACrBuffer.begin() + currentACrStart,
        [k] __device__(const CodewordT& cw) { return cw.isClassRepresentative(k.zero, k.free); });
    uint32_t len = thrust::distance(dACrBuffer.begin() + currentACrStart, acrEnd);
    assert(len == acrCache[ck]);
    zfColorsToACrLocations[k] = {currentACrStart, len};
    currentACrStart += len;
  }
  assert(currentACrStart == dACrBuffer.size());
  counters[Counters<SolverConfig>::ACR_COUNT] += zfColorsToACrLocations.size();

  // Finally, use the map to set the ARc location and length for every region, in device memory.
  thrust::host_vector<uint32_t> hACrStarts(dZFColors.size());
  thrust::host_vector<uint32_t> hACrLengths(dZFColors.size());
  thrust::host_vector<ZFColors> hZFColors(dZFColors);
  for (int i = tinyRegionCount; i < regionCount; i++) {
    auto& loc = zfColorsToACrLocations.at(hZFColors[i]);
    hACrStarts[i] = loc.start;
    hACrLengths[i] = loc.len;
  }
  dACrStarts = hACrStarts;
  dACrLengths = hACrLengths;
}
