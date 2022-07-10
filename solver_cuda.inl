// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <new>
#define CUB_STDERR
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/zip_function.h>

#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>
#include <cuda/barrier>
#include <vector>

#include "algos.hpp"
#include "codeword.hpp"

// CUDA implementation for playing all games at once
//
// TODO: this needs a lot of notes and docs consolidated
//
//
// This plays all games at once, playing a turn on each game and forming a set of next guesses for the next turn.
// Scoring all games subdivides those games with the same score into disjoint regions. We form an id for each region
// made up of a growing list of scores, and sorting the list of games by region id groups the regions together. Then we
// can find a single best guess for each region and play that for all games in the region.
//
// Region ids are chosen to ensure that a stable sort keeps games within each region in their original lexical order.
// Many algorithms pick the first lexical game on ties.
//
// Pseudocode for the algorithm:
//   start: all games get the same initial guess
//
//   while any games have guesses to play:
//     score all games against their next guess, if any, which was given per-region
//       append their score to their region id
//       if no games have guesses, then we're done
//     stable sort all games by region id
//     get start and run length for each region by id
//     for reach region:
//       games with a win at the end of their region id get no new guess
//       otherwise, find next guess using the region itself as the possible solutions set PS

// mmmfixme: name and placement of both of these
template <typename T, uint8_t WINNING_SCORE>
struct RegionIDLR {
  T value = 0;
  uint32_t index;

  __host__ __device__ RegionIDLR() : value(0), index(0) {}
  __host__ __device__ RegionIDLR(const RegionIDLR& r) : value(r.value), index(r.index) {}

  __host__ __device__ void append(const Score& s, int depth) {
    assert(depth < 16);
    value |= static_cast<T>(s.result) << (cuda::std::numeric_limits<T>::digits - (depth * CHAR_BIT));
  }

  __host__ __device__ void append(uint8_t s, int depth) {
    assert(depth < 16);
    value |= static_cast<T>(s) << (cuda::std::numeric_limits<T>::digits - (depth * CHAR_BIT));
  }

  __host__ __device__ bool isGameOver() const {
    auto v = value;
    while (v != 0) {
      if ((v & 0xFF) == WINNING_SCORE) return true;
      v >>= 8;
    }
    return false;
  }

  __host__ __device__ int countMovesPacked() const {
    auto v = value;
    int c = 0;
    while (v != 0) {
      c++;
      static constexpr auto highByteShift = cuda::std::numeric_limits<T>::digits - CHAR_BIT;
      if (((v & (static_cast<T>(0xFF) << highByteShift)) >> highByteShift) == WINNING_SCORE) break;
      v <<= 8;
    }
    return c;
  }

  __host__ __device__ void dump() const { printf("%016lx-%016lx\n", *(((uint64_t*)&value) + 1), *((uint64_t*)&value)); }

  std::ostream& dump(std::ostream& stream) const {
    std::ios state(nullptr);
    state.copyfmt(stream);
    stream << std::hex << std::setfill('0') << std::setw(16) << *(((uint64_t*)&value) + 1) << "-" << std::setw(16)
           << *((uint64_t*)&value);
    stream.copyfmt(state);
    return stream;
  }
};

#if 0
struct RegionIDRL {
  unsigned __int128 value = 0;
  uint32_t index;

  __host__ __device__ RegionIDRL() : value(0), index(0) {}
  __host__ __device__ RegionIDRL(const RegionIDRL& r) : value(r.value), index(r.index) {}

  __host__ __device__ void append(const Score& s, int depth) {
    assert(depth < 16);
    value = (value << 8) | s.result;
  }

  __host__ __device__ void append(uint8_t s, int depth) {
    assert(depth < 16);
    value = (value << 8) | s;
  }

  __host__ bool isFinal() const { return (value & 0xFF) == CodewordT::WINNING_SCORE.result; }
  __host__ __device__ bool isFinalPacked() const { return (value & 0xFF) == TOTAL_PACKED_SCORES - 1; }

  __host__ __device__ void dump() const { printf("%016lx-%016lx\n", *(((uint64_t*)&value) + 1), *((uint64_t*)&value)); }

  std::ostream& dump(std::ostream& stream) const {
    std::ios state(nullptr);
    state.copyfmt(stream);
    stream << std::hex << std::setfill('0') << std::setw(16) << *(((uint64_t*)&value) + 1) << "-" << std::setw(16)
           << *((uint64_t*)&value);
    stream.copyfmt(state);
    return stream;
  }
};
#endif

// using RegionID = RegionIDLR<uint64_t>;

template <typename T, uint8_t WINNING_SCORE>
std::ostream& operator<<(std::ostream& stream, const RegionIDLR<T, WINNING_SCORE>& r) {
  return r.dump(stream);
}

// scoring all games in a region subdivides it, one new region per score.
// - sorting the region by score gets us runs of games we can apply the same guess to

// region id: [s1, s2, s3, s4, ...] <-- sort by that, stable to keep lexical ordering

// - start: all games get the same initial guess
//
// - while any games have guesses to play:
//   - score all games against their next guess, if any, which was given per-region
//     - append their score to their region id
//     - if no games have guesses, then we're done
//   - sort all games by region id
//     - this re-shuffles within each region only
//   - get start and run length for each region by id
//     - at the start there are 14, at the end there are likely 1296
//   - for reach region:
//     - find next guess using the region as PS, this is the next guess for all games in this region
//       - games with a win at the end of their region id get no new guess

// for GPU
//
// - start:
//   - next_guesses[0..n] = IG
//   - PS[0..n] = AC[0..n]
//   - region_id[0..n] = {}
//
// - while true:
//   - grid to score PS[0..n] w/ next_guesses[0..n] => updated region_id[0..n]
//     - s = PS[i].score(next_guesses[i])
//     - s is in a local per-thread
//     - append s to region_id[i]
//   - reduce scores to a single, non-winning score, if any
//   - if no non-winning scores, break, we're done
//
//   - grid to sort PS by region_id
//
//   - grid to reduce PS to a set of regions
//     - list of region_id, start index, and length
//
//   - grid per-region to find next guess and update next_guesses
//     - regions with a win at the end get no new guess, -1ish

// or, for the last two:
//
//   - grid over PS
//     - first thread in region kicks off the work for finding the next guess for that region
//       - first thread if rid_i != rid_(i-1)
//     - when done, shares the ng w/ all threads in the region and they update next_guesses[0..n]
//     - trying to avoid the device-wide reduction and extra kernel kickoff's per-region

// The core of Knuth's Mastermind algorithm, and others, as CUDA compute kernels.
//
// Scores here are not the classic combination of black hits and white hits. A score's ordinal is (b(p + 1) -
// ((b - 1)b) / 2) + w. See docs/Score_Ordinals.md for details. By using the score's ordinal we can have densely packed
// set of counters to form the subset counts as we go. These scores never escape the GPU, so it doesn't matter that they
// don't match any other forms of scores in the rest of the program.

// Mastermind scoring function
//
// This mirrors the scalar version very closely. It's the full counting method from Knuth, plus some fun bit twiddling
// hacks and SWAR action. This is O(1) using warp SIMD intrinsics.
//
// Find black hits with xor, which leaves zero nibbles on matches, then count the zeros in the result. This is a
// variation on determining if a word has a zero byte from https://graphics.stanford.edu/~seander/bithacks.html. This
// part ends with using the GPU's SIMD popcount() to count the zero nibbles.
//
// Next, color counts come from the parallel buffer, and we can run over them and add up total hits, per Knuth[1], by
// aggregating min color counts between the secret and guess.

// TODO: early draft https://godbolt.org/z/ea7YjEPqf

template <uint PIN_COUNT>
__device__ uint scoreCodewords(const uint32_t secret, const uint4 secretColors, const uint32_t guess,
                               const uint4 guessColors) {
  constexpr uint unusedPinsMask = 0xFFFFFFFFu & ~((1lu << PIN_COUNT * 4u) - 1);
  uint v = secret ^ guess;  // Matched pins are now 0.
  v |= unusedPinsMask;      // Ensure that any unused pin positions are non-zero.
  uint r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint b = __popc(r);

  uint mins1 = __vminu4(secretColors.x, guessColors.x);
  uint mins2 = __vminu4(secretColors.y, guessColors.y);
  uint mins3 = __vminu4(secretColors.z, guessColors.z);
  uint mins4 = __vminu4(secretColors.w, guessColors.w);
  uint allHits = __vsadu4(mins1, 0);
  allHits += __vsadu4(mins2, 0);
  allHits += __vsadu4(mins3, 0);
  allHits += __vsadu4(mins4, 0);

  // Given w = ah - b, simplify to i = bp - ((b - 1)b) / 2) + ah. I wonder if the compiler noticed that.
  // https://godbolt.org/z/ab5vTn -- gcc 10.2 notices and simplifies, clang 11.0.0 misses it.
  return b * PIN_COUNT - ((b - 1) * b) / 2 + allHits;
}

// mmmfixme: which of these, if any, are screwed up due to the Thrust device vectors being properly sized, and not
//  rounded up to the next block size?

// Score all possible solutions against a given secret and compute subset sizes, which are the number of codewords per
// score.
template <typename SolverConfig, typename SubsetSizeT, typename CodewordT>
__device__ void computeSubsetSizes(SubsetSizeT* __restrict__ subsetSizes, const uint32_t secret,
                                   const uint4 secretColors, const CodewordT* __restrict__ regionIDsAsCodeword,
                                   uint32_t regionStart, uint32_t regionLength) {
  for (uint32_t i = regionStart; i < regionStart + regionLength; i++) {
    auto& ps = regionIDsAsCodeword[i];
    unsigned __int128 pc8 = ps.packedColors8();  // Annoying...
    uint score = scoreCodewords<SolverConfig::PIN_COUNT>(secret, secretColors, ps.packedCodeword(), *(uint4*)&pc8);
    SolverConfig::ALGO::accumulateSubsetSize(subsetSizes[score]);
  }
}

// Score all possible solutions against a given secret and compute subset sizes, which are the number of codewords per
// score. Specialized for small regions with a fixed length.
// TODO: more detailed specializations for specific small sizes, like 3.
template <typename SolverConfig, uint32_t REGION_LENGTH = 0, typename SubsetSizeT, typename CodewordT>
__device__ void computeSubsetSizesFixedLength(SubsetSizeT* __restrict__ subsetSizes, const uint32_t secret,
                                              const uint4 secretColors,
                                              const CodewordT* __restrict__ regionIDsAsCodeword, uint32_t regionStart,
                                              uint32_t secretIndex) {
  for (uint32_t i = regionStart; i < regionStart + REGION_LENGTH; i++) {
    if (i == secretIndex) {  // don't score w/ self, we know it's a win
      subsetSizes[SolverConfig::TOTAL_PACKED_SCORES - 1] = 1;
    } else {
      auto& ps = regionIDsAsCodeword[i];
      unsigned __int128 pc8 = ps.packedColors8();  // Annoying...
      uint score = scoreCodewords<SolverConfig::PIN_COUNT>(secret, secretColors, ps.packedCodeword(), *(uint4*)&pc8);
      subsetSizes[score] = 1;
    }
  }
}

// Keeps an index into the all codewords vector together with a rank on the GPU, and whether this codeword is a
// possible solution.
struct IndexAndRank {
  uint32_t index;
  uint32_t rank;
  bool isPossibleSolution;
  bool isFD;
};

// Reducer for per-thread guesses, used for CUB per-block and device reductions.
struct IndexAndRankReducer {
  __device__ __forceinline__ IndexAndRank operator()(const IndexAndRank& a, const IndexAndRank& b) const {
    // Always take the best rank. If it's a tie, take the one that could be a solution. If that's a tie, take lexically
    // first.
#if 0  // mmmfixme: worth it?
    if (b.isFD || a.isFD) {
      if (b.isFD && a.isFD) {
        if (b.isPossibleSolution ^ a.isPossibleSolution) return b.isPossibleSolution ? b : a;
        return (b.index < a.index) ? b : a;
      }
      return b.isFD ? b : a;
    }
#endif

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
  using CodewordT = Codeword<PIN_COUNT, COLOR_COUNT>;

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
  constexpr static uint32_t maxThreadsFromSubsetType() {
    uint32_t sharedMemSize = 48 * 1024;  // Default on 8.6
    uint32_t sharedMemPerThread = sizeof(T) * TOTAL_PACKED_SCORES;
    uint32_t threadsPerBlock = nextPowerOfTwo((sharedMemSize / sharedMemPerThread) / 2);
    return threadsPerBlock;
  }

  // How many threads will be put in each block. Always at least one warp, but no more than 512 (which needs to be tuned
  // more; 512 is picked based on results from 8p5c runs on MostParts and Knuth.)
  template <typename T>
  constexpr static uint32_t threadsPerBlock() {
    return std::clamp(std::min(static_cast<uint64_t>(maxThreadsFromSubsetType<T>()), CodewordT::TOTAL_CODEWORDS), 32ul,
                      512ul);
  }
  static constexpr uint32_t THREADS_PER_BLOCK = threadsPerBlock<SubsetSizeT>();

  // How many blocks we'll launch. This is rounded up to ensure we capture the last partial block. All kernels are
  // written to tolerate an incomplete final block.
  constexpr static uint32_t numBlocks(const uint32_t threadsPerBlock) {
    return (CodewordT::TOTAL_CODEWORDS + threadsPerBlock - 1) / threadsPerBlock;
  }
  static constexpr uint32_t NUM_BLOCKS = numBlocks(THREADS_PER_BLOCK);
  static constexpr uint32_t ROUNDED_TOTAL_CODEWORDS = NUM_BLOCKS * THREADS_PER_BLOCK;

  // These are the worst-case values over all types this config will be specialized with. Currently, those are 1, 2, and
  // 4 byte types. We use the most blocks with the largest type, but we need the most space for codewords with the
  // smallest type since the block size is larger, and we round up a full block.
  static constexpr uint32_t LARGEST_NUM_BLOCKS = numBlocks(threadsPerBlock<uint32_t>());
  static constexpr uint32_t LARGEST_ROUNDED_TOTAL_CODEWORDS =
      numBlocks(threadsPerBlock<uint8_t>()) * threadsPerBlock<uint8_t>();

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
static_assert(testConfig::numBlocks(testConfig::threadsPerBlock<uint32_t>()) == 1526);

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
                                      uint32_t nextMovesVecsSize, IndexAndRank* __restrict__ perBlockSolutions) {
  __shared__ typename SubsettingAlgosKernelConfig::SharedMemLayout sharedMem;

  const uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  auto subsetSizes = &sharedMem.subsetSizes[threadIdx.x * SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES];
  for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

  unsigned __int128 apc8 = allCodewords[tidGrid].packedColors8();  // Annoying...
  computeSubsetSizes<SubsettingAlgosKernelConfig::SolverConfig>(subsetSizes, allCodewords[tidGrid].packedCodeword(),
                                                                *(uint4*)&apc8, regionIDsAsCodeword, regionStart,
                                                                regionLength);

  auto possibleSolutionsCount = regionLength;
  bool isPossibleSolution = subsetSizes[SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES - 1] > 0;

  using ALGO = typename SubsettingAlgosKernelConfig::SolverConfig::ALGO;
  typename ALGO::RankingAccumulatorType rankingAccumulator{};
  uint32_t totalUsedSubsets = 0;
  for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) {
    if (subsetSizes[i] > 0) {
      ALGO::accumulateRanking(rankingAccumulator, subsetSizes[i], possibleSolutionsCount);
      totalUsedSubsets++;
    }
  }

  uint32_t rank = ALGO::computeRank(rankingAccumulator, possibleSolutionsCount);

  // A rank of 0 will prevent used or invalid codewords from being chosen.
  if (tidGrid >= SubsettingAlgosKernelConfig::CodewordT::TOTAL_CODEWORDS) {
    rank = 0;
    totalUsedSubsets = 0;
  } else {
    // Use the list of next moves sets to discard used codewords. nb: -1 to skip the new set.
    // TODO: I'd like to improve this. Ideally we wouldn't do this for low ranked guesses that won't be picked anyway.
    for (int i = 0; i < nextMovesVecsSize - 1; i++) {
      if (tidGrid == nextMovesVecs[i][regionIDsAsIndex[regionStart]]) {
        rank = 0;
        totalUsedSubsets = 0;
        break;
      }
    }
  }

  // Reduce to find the best solution we have in this block. This keeps the codeword index, rank, and possible solution
  // indicator together.
  __syncthreads();
  IndexAndRank iar{tidGrid, rank, isPossibleSolution, false && totalUsedSubsets == possibleSolutionsCount};
  IndexAndRank bestSolution =
      typename SubsettingAlgosKernelConfig::BlockReduce(sharedMem.reducerTmpStorage).Reduce(iar, IndexAndRankReducer());

  if (threadIdx.x == 0) {
    perBlockSolutions[blockIdx.x] = bestSolution;
  }

  // mmmfixme
  // If we find some guesses which are fully discriminating, we want to pick the first one lexically to play. tidGrid is
  // the same as the ordinal for each member of allCodewords, so we can simply take the min tidGrid.
  //  if (possibleSolutionsCount <= SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES) {
  //    if (totalUsedSubsets == possibleSolutionsCount) {
  // I don't really like this, but it's tested out faster than doing a per-block reduction and a subsequent
  // device-wide reduction, like for the index and score above. Likewise, doing a warp-level reduction centered
  // around __reduce_min_sync() tests the same speed as just the atomicMin().
  //      atomicMin(&(littleStuff->fdGuess), tidGrid);
  //    }
  //  }
}

// Reduce the per-block best guesses from subsettingAlgosKernel to generate a single, best guess. This is then set
// as the next move for the region.
template <uint32_t blockSize>
__global__ void reduceBestGuess(IndexAndRank* __restrict__ perBlockSolutions, const uint32_t solutionsCount,
                                const uint32_t* __restrict__ regionIDsAsIndex, uint32_t* __restrict__ nextMoves,
                                const int regionStart, const int regionLength) {
  uint32_t idx = threadIdx.x;
  IndexAndRankReducer reduce;
  IndexAndRank bestGuess{0, 0, false, false};
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
                                       IndexAndRank* __restrict__ perBlockSolutions) {
  subsettingAlgosKernel<SubsettingAlgosKernelConfig>
      <<<SubsettingAlgosKernelConfig::NUM_BLOCKS, SubsettingAlgosKernelConfig::THREADS_PER_BLOCK>>>(
          allCodewords, regionIDsAsCodeword, regionIDsAsIndex, regionStart, regionLength, nextMovesVecs,
          nextMovesVecsSize, perBlockSolutions);
  CubDebug(cudaGetLastError());

  // nb: block size on this one must be a power of 2
  reduceBestGuess<128><<<1, 128>>>(perBlockSolutions, SubsettingAlgosKernelConfig::NUM_BLOCKS, regionIDsAsIndex,
                                   nextMoves, regionStart, regionLength);
  CubDebug(cudaGetLastError());
}

// Holds all the constants we need to kick off the CUDA kernel for all the fully discriminating optimization given a
// solver config.
template <typename SolverConfig_>
struct FDOptKernelConfig {
  using SolverConfig = SolverConfig_;
  static constexpr uint8_t PIN_COUNT = SolverConfig::PIN_COUNT;
  static constexpr uint8_t COLOR_COUNT = SolverConfig::COLOR_COUNT;
  static constexpr bool LOG = SolverConfig::LOG;
  using CodewordT = Codeword<PIN_COUNT, COLOR_COUNT>;

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
// codeword can fully discriminate all of the possible solutions (i.e., it produces a different score for each one),
// then play it right away since it will tell us the winner.
//
// This compares PS with itself looking for a fully discriminating guess, and falls back to the full algo if none is
// found.
//
// This is an interesting shortcut. It doesn't change the results of any of the subsetting algorithms at all: average
// turns, max turns, max secret, and the full histograms all remain precisely the same. What does change is the number
// of scores computed, and the runtime.
//
// nb: one block, one warp for this one. Max region length is 45 for 8 pin games, which is our pin max, so fewer than
// half the threads even have to loop to pickup all the data and we get away with a single warp reduction.

template <typename FDOptKernelConfig, typename SubsettingAlgosKernelConfig, typename CodewordT>
__global__ void fullyDiscriminatingOpt(const CodewordT* __restrict__ allCodewords,
                                       const CodewordT* __restrict__ regionIDsAsCodeword,
                                       const uint32_t* __restrict__ regionIDsAsIndex, uint32_t regionStart,
                                       uint32_t regionLength, uint32_t* __restrict__ nextMoves,
                                       uint32_t** __restrict__ nextMovesVecs, uint32_t nextMovesVecsSize,
                                       IndexAndRank* __restrict__ perBlockSolutions) {
  assert(blockIdx.x == 0);   // Single block
  assert(blockDim.x == 32);  // Single warp

  using SolverConfig = typename FDOptKernelConfig::SolverConfig;
  __shared__ typename FDOptKernelConfig::SharedMemLayout sharedMem;
  uint result = cuda::std::numeric_limits<uint>::max();

  for (uint idx = threadIdx.x; idx < regionLength; idx += blockDim.x) {
    auto subsetSizes = &sharedMem.subsetSizes[idx * SolverConfig::TOTAL_PACKED_SCORES];
    for (int i = 0; i < SolverConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

    unsigned __int128 apc8 = regionIDsAsCodeword[idx + regionStart].packedColors8();  // Annoying...

    // mmmfixme: need to tune these small specializations and see which ones are really necessary, which ones should be
    //  specialized by hand, etc.
    if (regionLength == 3) {
      computeSubsetSizesFixedLength<SolverConfig, 3>(
          subsetSizes, regionIDsAsCodeword[idx + regionStart].packedCodeword(), *(uint4*)&apc8, regionIDsAsCodeword,
          regionStart, idx + regionStart);
    } else if (regionLength == 4) {
      computeSubsetSizesFixedLength<SolverConfig, 4>(
          subsetSizes, regionIDsAsCodeword[idx + regionStart].packedCodeword(), *(uint4*)&apc8, regionIDsAsCodeword,
          regionStart, idx + regionStart);
    } else if (regionLength == 5) {
      computeSubsetSizesFixedLength<SolverConfig, 5>(
          subsetSizes, regionIDsAsCodeword[idx + regionStart].packedCodeword(), *(uint4*)&apc8, regionIDsAsCodeword,
          regionStart, idx + regionStart);
    } else {
      computeSubsetSizes<SolverConfig>(subsetSizes, regionIDsAsCodeword[idx + regionStart].packedCodeword(),
                                       *(uint4*)&apc8, regionIDsAsCodeword, regionStart, regionLength);
    }

    uint32_t totalUsedSubsets = 0;
    for (int i = 0; i < SolverConfig::TOTAL_PACKED_SCORES; i++) {
      if (subsetSizes[i] > 0) {
        totalUsedSubsets++;
      }
    }

    if (totalUsedSubsets == regionLength) {
      result = min(result, idx + regionStart);
    }
  }

  __syncthreads();
  uint bestSolution =
      typename FDOptKernelConfig::SmallOptsBlockReduce(sharedMem.smallOptsReducerTmpStorage).Reduce(result, cub::Min());

  if (threadIdx.x == 0) {
    if (bestSolution < cuda::std::numeric_limits<uint>::max()) {
      for (int i = 0; i < regionLength; i++) {
        nextMoves[regionIDsAsIndex[i + regionStart]] = regionIDsAsIndex[bestSolution];
      }
    } else {
      // Fallback on the big kernel
      launchSubsettingKernel<SubsettingAlgosKernelConfig>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex,
                                                          nextMoves, regionStart, regionLength, nextMovesVecs,
                                                          nextMovesVecsSize, perBlockSolutions);
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
                                    uint32_t nextMovesVecsSize, IndexAndRank* __restrict__ perBlockSolutionsPool) {
  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidGrid < regionCount) {
    auto regionStart = regionStarts[offset + tidGrid];
    auto regionLength = regionLengths[offset + tidGrid];
    auto perBlockSolutions =
        &perBlockSolutionsPool[SubsettingAlgosKernelConfig<SolverConfig, uint32_t>::LARGEST_NUM_BLOCKS * tidGrid];

    using config8 = SubsettingAlgosKernelConfig<SolverConfig, uint8_t>;
    using config16 = SubsettingAlgosKernelConfig<SolverConfig, uint16_t>;
    using config32 = SubsettingAlgosKernelConfig<SolverConfig, uint32_t>;

    if (config8::shouldUseType(regionLength)) {
      if (regionLength < SolverConfig::TOTAL_PACKED_SCORES) {
        using configFDOpt = FDOptKernelConfig<SolverConfig>;
        fullyDiscriminatingOpt<configFDOpt, config8><<<1, configFDOpt::THREADS_PER_BLOCK>>>(
            allCodewords, regionIDsAsCodeword, regionIDsAsIndex, regionStart, regionLength, nextMoves, nextMovesVecs,
            nextMovesVecsSize, perBlockSolutions);
      } else {
        launchSubsettingKernel<config8>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart,
                                        regionLength, nextMovesVecs, nextMovesVecsSize, perBlockSolutions);
      }
    } else if (config16::shouldUseType(regionLength)) {
      launchSubsettingKernel<config16>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart,
                                       regionLength, nextMovesVecs, nextMovesVecsSize, perBlockSolutions);
    } else {
      launchSubsettingKernel<config32>(allCodewords, regionIDsAsCodeword, regionIDsAsIndex, nextMoves, regionStart,
                                       regionLength, nextMovesVecs, nextMovesVecsSize, perBlockSolutions);
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

// mmmfixme: docs
//  - Need to outline the key parts of the translation to GPU
//  - This uses a lot of device memory right now. Need to figure out the max size game as is and go from there.
//    - the pins of a packed codeword are 32bits. Could drop the colors and re-compute them as needed on-device.
//    - could make a packed regionID w/ 6bit scores
//    - region starts and lengths could be delta coded and variable size, etc.
//  - Much of this work is a serial list of Thrust or kernels and could be a parallel graph, but the time spent outside
//    of the main subsetting kernel is a tiny fraction of the overall work right now, so keeping it simple.

template <typename SolverConfig>
void SolverCUDA<SolverConfig>::playAllGames(uint32_t packedInitialGuess) {
  constexpr static bool LOG = SolverConfig::LOG;
  using RegionID = RegionIDLR<unsigned __int128, SolverConfig::TOTAL_PACKED_SCORES - 1>;

  thrust::device_vector<CodewordT> dAllCodewords = CodewordT::getAllCodewords();

  // mmmfixme: how to do this with a thrust vector?
  //  CubDebugExit(cudaMemAdvise(thrust::raw_pointer_cast(dAllCodewords.data()), sizeof(CodewordT) *
  //  allCodewords.size(), cudaMemAdviseSetReadMostly, 0));

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

  // Starting case: all games, initial guess.
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

  // Space for the intermediate reduction results out of the subsetting algos kernel.
  // mmmfixme: we need a set per concurrent kernel
  size_t chunkSize = 256;  // mmmfixme: placement
  thrust::device_vector<IndexAndRank> dPerBlockSolutions(
      SubsettingAlgosKernelConfig<SolverConfig, uint32_t>::LARGEST_NUM_BLOCKS * chunkSize);

  thrust::device_vector<uint32_t> dRegionIDsAsIndex(dRegionIDs.size());
  thrust::device_vector<CodewordT> dRegionIDsAsCodeword(dRegionIDs.size());

  int depth = 0;
  auto dRegionIDsEnd = dRegionIDs.end();  // The set of active games contracts as we go

  while (true) {
    // mmmfixme: tmp
    auto startTime = chrono::high_resolution_clock::now();
    depth++;

    if (LOG) printf("\nDepth = %d\n", depth);

    // Score all games against their next guess, if any, which was given per-region. Append the score to the game's
    // region id.
    auto pdAllCodewords = thrust::raw_pointer_cast(dAllCodewords.data());
    thrust::for_each(
        dRegionIDs.begin(), dRegionIDsEnd, [depth, pdAllCodewords, pdNextMoves] __device__(RegionID & regionID) {
          if (!regionID.isGameOver()) {
            auto cwi = regionID.index;
            unsigned __int128 apc8 = pdAllCodewords[cwi].packedColors8();               // Annoying...
            unsigned __int128 npc8 = pdAllCodewords[pdNextMoves[cwi]].packedColors8();  // Annoying...
            uint8_t s = scoreCodewords<SolverConfig::PIN_COUNT>(pdAllCodewords[cwi].packedCodeword(), *(uint4*)&apc8,
                                                                pdAllCodewords[pdNextMoves[cwi]].packedCodeword(),
                                                                *(uint4*)&npc8);
            regionID.append(s, depth);
          }
        });

    // Push won games to the end and focus on the remaining games
    dRegionIDsEnd = thrust::partition(dRegionIDs.begin(), dRegionIDsEnd,
                                      [] __device__(const RegionID& r) { return !r.isGameOver(); });

    if (LOG) printf("Number of games left: %ld\n", dRegionIDsEnd - dRegionIDs.begin());

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
    if (LOG) printf("Number of regions: %lu\n", regionCount);

    // Now build starts for each region
    thrust::exclusive_scan(dRegionLengths.begin(), dRegionLengths.begin() + regionCount, dRegionStarts.begin());

    // Sort the regions by length. Lets us batch up work for regions of different interesting sizes below
    thrust::sort_by_key(dRegionLengths.begin(), dRegionLengths.begin() + regionCount, dRegionStarts.begin());

    // Have to take the hit and pull the region lengths back, so we can launch different kernels
    // TODO: would be nice to have this async with other work above, not needed until later
    hRegionLengths = dRegionLengths;

    // TODO: these could probably be one zipped transform
    // TODO: re-test these. Trades a lot of device space for a small time gain, worth it?
    thrust::transform(dRegionIDs.begin(), dRegionIDsEnd, dRegionIDsAsIndex.begin(),
                      [] __device__(const RegionID& r) { return r.index; });
    thrust::transform(dRegionIDs.begin(), dRegionIDsEnd, dRegionIDsAsCodeword.begin(),
                      [pdAllCodewords] __device__(const RegionID& r) { return pdAllCodewords[r.index]; });

    if (LOG) {  // mmmfixme: tmp, factor if I decide to keep it
      auto endTime = chrono::high_resolution_clock::now();
      chrono::duration<float, milli> elapsedMS = endTime - startTime;
      auto elapsedS = elapsedMS.count() / 1000;
      cout << "P1 elapsed time " << commaString(elapsedS) << "s" << endl;
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

    uint32_t tinyRegionCount = 0;
    uint32_t tinyGameCount = 0;
    while (tinyRegionCount < regionCount && hRegionLengths[tinyRegionCount] <= 2) {
      tinyGameCount += hRegionLengths[tinyRegionCount];
      tinyRegionCount++;
    }
    if (tinyRegionCount > 0) {
      // mmmfixme: error check the grid launches, even the nested ones, especially for
      //  cudaErrorLaunchPendingCountExceeded
      //  - default is 2048 I think.
      //  - check and adjust as necessary for our sub-kernels
      //    - cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);
      //  - https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/
      nextGuessTiny<<<1, 128>>>(pdRegionsAsIndex, pdNextMoves, pdRegionStarts, pdRegionLengths, tinyRegionCount);
    }

    if (LOG) {
      printf("Tiny regions: %d, totalling %d games\n", tinyRegionCount, tinyGameCount);

      uint32_t fdRegionCount = 0;
      uint32_t fdGameCount = 0;
      for (uint32_t k = tinyRegionCount; k < regionCount && hRegionLengths[k] < SolverConfig::TOTAL_PACKED_SCORES;
           k++) {
        fdGameCount += hRegionLengths[k];
        fdRegionCount++;
      }
      printf("Possibly fully discriminating regions: %d, totalling %d games\n", fdRegionCount, fdGameCount);
    }

    int bigBoyLaunches = 0;
    for (size_t offset = tinyRegionCount; offset < regionCount; offset += chunkSize) {
      auto regionsToDo = min(chunkSize, regionCount - offset);
      int threadsPerBlock = 4;  // mmmfixme: odd... needs tuning and better blocking

      bigBoyLaunches++;
      auto pdPerBlockSolutions = thrust::raw_pointer_cast(dPerBlockSolutions.data());
      nextGuessForRegions<SolverConfig><<<(regionsToDo + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
          pdAllCodewords, pdRegionsAsCodeword, pdRegionsAsIndex, pdNextMoves, pdRegionStarts, pdRegionLengths, offset,
          regionsToDo, pdNextMovesVecs, nextMovesVecsSize, pdPerBlockSolutions);
    }

    if (LOG) printf("Big Boy Launches: %d\n", bigBoyLaunches);
    CubDebug(cudaDeviceSynchronize());

    if (LOG) {  // mmmfixme: tmp, factor if I decide to keep it
      auto endTime = chrono::high_resolution_clock::now();
      chrono::duration<float, milli> elapsedMS = endTime - startTime;
      auto elapsedS = elapsedMS.count() / 1000;
      cout << "P2 elapsed time " << commaString(elapsedS) << "s" << endl;
      startTime = chrono::high_resolution_clock::now();
    }

    if (depth == MAX_SUPPORTED_TURNS) {
      printf("\nMax depth reached, impl is broken!\n");
      break;
    }
  }

  if (LOG) cout << "Last actual depth: " << depth << endl;

  // Post-process for stats
  hRegionIDs = dRegionIDs;
  for (int i = 0; i < hRegionIDs.size(); i++) {
    auto c = hRegionIDs[i].countMovesPacked();
    this->maxDepth = max<size_t>(this->maxDepth, c);
    this->totalTurns += c;
  }
}
