//
// Created by Michael Magruder on 6/30/22.
//

#include "new_algo.h"

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
#include <new>
#include <vector>

#include "codeword.hpp"
#include "cuda_gpu_interface.cuh"
#include "preset_initial_guesses.h"
#include "strategy_config.hpp"

constexpr static int PIN_COUNT = 8;
constexpr static int COLOR_COUNT = 5;
static constexpr uint32_t INITIAL_GUESS = presetInitialGuessKnuth<PIN_COUNT, COLOR_COUNT>();
using CodewordT = Codeword<PIN_COUNT, COLOR_COUNT>;
constexpr static uint32_t MAX_SCORE_SLOTS = (PIN_COUNT << 4u) + 1;
static constexpr int TOTAL_PACKED_SCORES = ((PIN_COUNT * (PIN_COUNT + 3)) / 2) + 1;

using SC = SubsettingStrategyConfig<PIN_COUNT, COLOR_COUNT, false, Algo::Knuth, uint32_t>;

struct LittleStuffNew {
  IndexAndScore bestGuess;
  //  uint32_t fdGuess;
  //  uint32_t usedCodewordsCount;
  //  uint32_t usedCodewords[100];
};

static CodewordT nextGuess(const vector<CodewordT>& possibleSolutions, const vector<CodewordT>& usedCodewords) {
  CodewordT bestGuess;
  size_t bestScore = 0;
  bool bestIsPossibleSolution = false;
  auto& allCodewords = CodewordT::getAllCodewords();
  int subsetSizes[MAX_SCORE_SLOTS];
  fill(begin(subsetSizes), end(subsetSizes), 0);
  for (const auto& g : allCodewords) {
    bool isPossibleSolution = false;
    for (const auto& ps : possibleSolutions) {
      Score r = g.score(ps);
      subsetSizes[r.result]++;
      if (r == CodewordT::WINNING_SCORE) {
        isPossibleSolution = true;  // Remember if this guess is in the set of possible solutions
      }
    }

    //    size_t score = computeSubsetScore();
    int largestSubsetSize = 0;  // Maximum number of codewords that could be retained by using this guess
    for (auto& s : subsetSizes) {
      if (s > largestSubsetSize) {
        largestSubsetSize = s;
      }
      s = 0;
    }

    // Invert largestSubsetSize, and return the minimum number of codewords that could be eliminated by using this guess
    int score = possibleSolutions.size() - largestSubsetSize;

    if (score > bestScore || (!bestIsPossibleSolution && isPossibleSolution && score == bestScore)) {
      if (find(usedCodewords.cbegin(), usedCodewords.cend(), g) != usedCodewords.end()) {
        continue;  // Ignore codewords we've already used
      }
      bestScore = score;
      bestGuess = g;
      bestIsPossibleSolution = isPossibleSolution;
    }
  }
  return bestGuess;
}

template <typename T>
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

  __host__ bool isFinal() const {
    auto v = value;
    while (v != 0) {
      if ((v & 0xFF) == CodewordT::WINNING_SCORE.result) return true;
      v >>= 8;
    }
    return false;
  }

  __host__ __device__ bool isFinalPacked() const {
    auto v = value;
    while (v != 0) {
      if ((v & 0xFF) == TOTAL_PACKED_SCORES - 1) return true;
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
      if (((v & (static_cast<T>(0xFF) << highByteShift)) >> highByteShift) == TOTAL_PACKED_SCORES - 1) break;
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

using RegionID = RegionIDLR<unsigned __int128>;
// using RegionID = RegionIDLR<uint64_t>;

std::ostream& operator<<(std::ostream& stream, const RegionID& r) { return r.dump(stream); }

struct RegionRun {
  RegionID region;
  int start;
  int length;
};

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

// mmmfixme: fix names to possibleSolutions instead of regions/runs.
template <uint32_t PIN_COUNT, Algo ALGO, typename SubsetSizeT, typename CodewordT>
__device__ void computeSubsetSizesNew(SubsetSizeT* __restrict__ subsetSizes, const uint32_t secret,
                                      const uint4 secretColors, const CodewordT* __restrict__ regions,
                                      uint32_t runStart, uint32_t runLength) {
  for (uint32_t i = runStart; i < runStart + runLength; i++) {
    auto& ps = regions[i];
    unsigned __int128 pc8 = ps.packedColors8();  // Annoying...
    uint score = scoreCodewords<PIN_COUNT>(secret, secretColors, ps.packedCodeword(), *(uint4*)&pc8);
    if (ALGO == Algo::MostParts) {
      subsetSizes[score] = 1;
    } else {
      subsetSizes[score]++;
    }
  }
}

template <typename SubsettingAlgosKernelConfig, typename LittleStuffT>
__global__ void subsettingAlgosKernelNew(
    const typename SubsettingAlgosKernelConfig::CodewordT* __restrict__ allCodewords,
    const typename SubsettingAlgosKernelConfig::CodewordT* __restrict__ regions, uint32_t runStart, uint32_t runLength,
    LittleStuffT* __restrict__ littleStuff, IndexAndScore* __restrict__ perBlockSolutions) {
  __shared__ typename SubsettingAlgosKernelConfig::SharedMemLayout sharedMem;

  const uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  auto subsetSizes = &sharedMem.subsetSizes[threadIdx.x * SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES];
  for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

  unsigned __int128 apc8 = allCodewords[tidGrid].packedColors8();  // Annoying...
  computeSubsetSizesNew<SubsettingAlgosKernelConfig::PIN_COUNT, SubsettingAlgosKernelConfig::ALGO>(
      subsetSizes, allCodewords[tidGrid].packedCodeword(), *(uint4*)&apc8, regions, runStart, runLength);

  auto possibleSolutionsCount = runLength;
  bool isPossibleSolution = subsetSizes[SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES - 1] > 0;

  uint32_t largestSubsetSize = 0;
  uint32_t totalUsedSubsets = 0;
  float entropySum = 0.0;
  float expectedSize = 0.0;
  for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) {
    if (subsetSizes[i] > 0) {
      totalUsedSubsets++;
      switch (SubsettingAlgosKernelConfig::ALGO) {
        case Knuth:
          largestSubsetSize = max(largestSubsetSize, subsetSizes[i]);
          break;
        case MostParts:
          // Already done
          break;
        case ExpectedSize:
          expectedSize += ((float)subsetSizes[i] * (float)subsetSizes[i]) / possibleSolutionsCount;
          break;
        case Entropy:
          float pi = (float)subsetSizes[i] / possibleSolutionsCount;
          entropySum -= pi * log(pi);
          break;
      }
    }
  }
  uint32_t score;
  switch (SubsettingAlgosKernelConfig::ALGO) {
    case Knuth:
      score = possibleSolutionsCount - largestSubsetSize;
      break;
    case MostParts:
      score = totalUsedSubsets;
      break;
    case ExpectedSize:
#pragma nv_diagnostic push
#pragma nv_diag_suppress 68
      // This is a bit broken, and needs to be to match the semantics in the paper.
      score = (uint32_t)round(expectedSize * 1'000'000.0) * -1;  // 9 digits of precision
#pragma nv_diagnostic pop
      break;
    case Entropy:
      score = round(entropySum * 1'000'000'000.0);  // 9 digits of precision
      break;
  }

  // A score of 0 will prevent used or invalid codewords from being chosen.
  if (tidGrid >= SubsettingAlgosKernelConfig::CodewordT::TOTAL_CODEWORDS) {
    score = 0;
    totalUsedSubsets = 0;
  }
  // mmmfixme:
  //  for (int i = 0; i < littleStuff->usedCodewordsCount; i++) {
  //    if (allCodewords[tidGrid] == littleStuff->usedCodewords[i]) score = 0;
  //  }

  // Reduce to find the best solution we have in this block. This keeps the codeword index, score, and possible solution
  // indicator together.
  __syncthreads();
  IndexAndScore ias{tidGrid, score, isPossibleSolution, false && totalUsedSubsets == possibleSolutionsCount};
  IndexAndScore bestSolution = typename SubsettingAlgosKernelConfig::BlockReduce(sharedMem.reducerTmpStorage)
                                   .Reduce(ias, IndexAndScoreReducer());

  if (threadIdx.x == 0) {
    perBlockSolutions[blockIdx.x] = bestSolution;
  }

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

template <uint32_t blockSize>
__global__ void reduceMaxScoreNew(IndexAndScore* __restrict__ perBlockSolutions, const uint32_t solutionsCount,
                                  const uint32_t* __restrict__ regions, int* __restrict__ nextMoves, const int runStart,
                                  const int runLength) {
  uint32_t idx = threadIdx.x;
  IndexAndScoreReducer reduce;
  IndexAndScore bestScore{0, 0, false, false};
  for (uint32_t i = idx; i < solutionsCount; i += blockSize) {
    bestScore = reduce(bestScore, perBlockSolutions[i]);
  }

  __shared__ IndexAndScore shared[blockSize];
  shared[idx] = bestScore;
  __syncthreads();

  for (uint32_t size = blockSize / 2; size > 0; size /= 2) {
    if (idx < size) {
      shared[idx] = reduce(shared[idx], shared[idx + size]);
    }
    __syncthreads();
  }

  for (uint32_t i = idx; i < runLength; i += blockSize) {
    nextMoves[regions[i + runStart]] = shared[0].index;
  }
}

__global__ void simpleRuns(const uint32_t* __restrict__ regions, int* __restrict__ nextMoves,
                           const int* __restrict__ runStarts, const int* __restrict__ runLengths, const int runCount) {
  uint32_t idx = threadIdx.x;

  for (int runIndex = idx; runIndex < runCount; runIndex += blockDim.x) {
    auto runStart = runStarts[runIndex];
    auto runLength = runLengths[runIndex];
    if (runLength == 1) {
      nextMoves[regions[runStart]] = regions[runStart];
    } else if (runLength == 2) {
      nextMoves[regions[runStart]] = regions[runStart];
      nextMoves[regions[runStart + 1]] = regions[runStart];
    }
  }
}

template <typename SubsettingAlgosKernelConfig>
__global__ void smallPSOpt(const typename SubsettingAlgosKernelConfig::CodewordT* __restrict__ regionsAsCodewords,
                           const uint32_t* __restrict__ regionsAsIndex, uint32_t runStart, uint32_t runLength,
                           int* __restrict__ nextMoves) {
  __shared__ typename SubsettingAlgosKernelConfig::SharedMemLayout sharedMem;

  const uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  uint result = cuda::std::numeric_limits<uint>::max();

  if (tidGrid < runLength) {
    auto subsetSizes = &sharedMem.subsetSizes[threadIdx.x * SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES];
    for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

    unsigned __int128 apc8 = regionsAsCodewords[tidGrid + runStart].packedColors8();  // Annoying...
    computeSubsetSizesNew<SubsettingAlgosKernelConfig::PIN_COUNT, Algo::MostParts>(
        subsetSizes, regionsAsCodewords[tidGrid + runStart].packedCodeword(), *(uint4*)&apc8, regionsAsCodewords,
        runStart, runLength);

    uint32_t totalUsedSubsets = 0;
    for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) {
      if (subsetSizes[i] > 0) {
        totalUsedSubsets++;
      }
    }

    if (totalUsedSubsets == runLength) {
      result = tidGrid + runStart;
    }
  }

  __syncthreads();
  uint bestSolution = typename SubsettingAlgosKernelConfig::SmallOptsBlockReduce(sharedMem.smallOptsReducerTmpStorage)
                          .Reduce(result, cub::Min());

  if (threadIdx.x == 0 && bestSolution < cuda::std::numeric_limits<uint>::max()) {
    for (int i = 0; i < runLength; i++) {
      nextMoves[regionsAsIndex[i + runStart]] = regionsAsIndex[bestSolution];
    }
  }
}

__global__ void newGuessForRegions(const CodewordT* __restrict__ allCodewords,
                                   const CodewordT* __restrict__ regionsAsCodeword,
                                   const uint32_t* __restrict__ regionsAsIndex, int* __restrict__ nextMoves,
                                   const int* __restrict__ runStarts, const int* __restrict__ runLengths,
                                   const int offset, const int runsCount,
                                   IndexAndScore* __restrict__ perBlockSolutionsPool) {
  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidGrid < runsCount) {
    auto runStart = runStarts[offset + tidGrid];
    auto runLength = runLengths[offset + tidGrid];
    auto perBlockSolutions =
        &perBlockSolutionsPool[SubsettingAlgosKernelConfig<SC, uint32_t>::LARGEST_NUM_BLOCKS * tidGrid];

    using config8 = SubsettingAlgosKernelConfig<SC, uint8_t>;
    using config16 = SubsettingAlgosKernelConfig<SC, uint16_t>;
    using config32 = SubsettingAlgosKernelConfig<SC, uint32_t>;

    if (config8::shouldUseType(runLength)) {
      if (runLength < TOTAL_PACKED_SCORES) {
        nextMoves[regionsAsIndex[runStart]] = -1;
        smallPSOpt<config8>
            <<<1, config8::THREADS_PER_BLOCK>>>(regionsAsCodeword, regionsAsIndex, runStart, runLength, nextMoves);
        CubDebug(cudaDeviceSynchronize());
        if (nextMoves[regionsAsIndex[runStart]] != -1) {
          return;
        }
      }

      subsettingAlgosKernelNew<config8, LittleStuffNew><<<config8::NUM_BLOCKS, config8::THREADS_PER_BLOCK>>>(
          allCodewords, regionsAsCodeword, runStart, runLength, nullptr, perBlockSolutions);
      CubDebug(cudaGetLastError());

      // nb: block size on this one must be a power of 2
      reduceMaxScoreNew<128>
          <<<1, 128>>>(perBlockSolutions, config8::NUM_BLOCKS, regionsAsIndex, nextMoves, runStart, runLength);
      CubDebug(cudaGetLastError());
    } else if (config16::shouldUseType(runLength)) {
      subsettingAlgosKernelNew<config16, LittleStuffNew><<<config16::NUM_BLOCKS, config16::THREADS_PER_BLOCK>>>(
          allCodewords, regionsAsCodeword, runStart, runLength, nullptr, perBlockSolutions);
      CubDebug(cudaGetLastError());

      // nb: block size on this one must be a power of 2
      reduceMaxScoreNew<128>
          <<<1, 128>>>(perBlockSolutions, config8::NUM_BLOCKS, regionsAsIndex, nextMoves, runStart, runLength);
      CubDebug(cudaGetLastError());
    } else {
      subsettingAlgosKernelNew<config32, LittleStuffNew><<<config32::NUM_BLOCKS, config32::THREADS_PER_BLOCK>>>(
          allCodewords, regionsAsCodeword, runStart, runLength, nullptr, perBlockSolutions);
      CubDebug(cudaGetLastError());

      // nb: block size on this one must be a power of 2
      reduceMaxScoreNew<128>
          <<<1, 128>>>(perBlockSolutions, config8::NUM_BLOCKS, regionsAsIndex, nextMoves, runStart, runLength);
      CubDebug(cudaGetLastError());
    }
  }
}

void new_algo::runGPU() {
  auto gpuInterface = new CUDAGPUInterface<SC>(CodewordT::getAllCodewords());

  vector<CodewordT>& allCodewords = CodewordT ::getAllCodewords();

  thrust::host_vector<CodewordT> hAllCodewords = allCodewords;
  thrust::device_vector<CodewordT> dAllCodewords = hAllCodewords;

  // mmmfixme: how to do this with a thrust vector?
  //  CubDebugExit(cudaMemAdvise(thrust::raw_pointer_cast(dAllCodewords.data()), sizeof(CodewordT) *
  //  allCodewords.size(), cudaMemAdviseSetReadMostly, 0));

  // Starting case: all games, initial guess.
  //  vector<CodewordT> used{};
  int igIndex = 0;
  for (; igIndex < allCodewords.size(); igIndex++) {
    if (allCodewords[igIndex] == INITIAL_GUESS) break;
  }
  thrust::device_vector<int> dNextMoves(allCodewords.size(), igIndex);
  thrust::host_vector<RegionID> hRegions(allCodewords.size());
  for (int i = 0; i < hRegions.size(); i++) hRegions[i].index = i;
  thrust::device_vector<RegionID> dRegions = hRegions;

  //  thrust::device_vector<RegionID> runIds(dRegions.size());
  thrust::device_vector<int> runStarts(dRegions.size());
  thrust::device_vector<int> runLengths(dRegions.size());
  //  thrust::host_vector<int> hRunStarts(dRegions.size());
  thrust::host_vector<int> hRunLengths(dRegions.size());

  size_t chunkSize = 256;
  //  thrust::device_vector<LittleStuffNew> dLittleStuffs(chunkSize);
  thrust::device_vector<IndexAndScore> dPerBlockSolutions(
      SubsettingAlgosKernelConfig<SC, uint32_t>::LARGEST_NUM_BLOCKS * chunkSize);
  //  auto pdLittleStuffs = thrust::raw_pointer_cast(dLittleStuffs.data());
  auto pdPerBlockSolutions = thrust::raw_pointer_cast(dPerBlockSolutions.data());

  thrust::device_vector<uint32_t> dRegionsAsIndex(dRegions.size());
  thrust::device_vector<CodewordT> dRegionsAsCodeword(dRegions.size());

  auto dRegionsEnd = dRegions.end();

  int depth = 0;

  // If no games have new moves, then we're done
  bool anyNewMoves = true;
  while (anyNewMoves) {
    auto startTime = chrono::high_resolution_clock::now();

    depth++;
    printf("\n---------- depth = %d ----------\n\n", depth);

    // mmmfixme: need a vector of dNextMoves, one per level, to keep history to build the strategy afterwards.
    //  - should also allow the "used codewords" set per-game, for algos that require it.

    auto pAllCodewords = thrust::raw_pointer_cast(dAllCodewords.data());
    auto pNextmoves = thrust::raw_pointer_cast(dNextMoves.data());
    //    auto pRegions = thrust::raw_pointer_cast(dRegions.data());

    thrust::for_each(dRegions.begin(), dRegionsEnd, [=] __device__(RegionID & r) {
      auto cwi = r.index;
      //      if (pNextmoves[cwi] != -1) {
      if (!r.isFinalPacked()) {
        unsigned __int128 apc8 = pAllCodewords[cwi].packedColors8();              // Annoying...
        unsigned __int128 npc8 = pAllCodewords[pNextmoves[cwi]].packedColors8();  // Annoying...
        uint8_t s = scoreCodewords<PIN_COUNT>(pAllCodewords[cwi].packedCodeword(), *(uint4*)&apc8,
                                              pAllCodewords[pNextmoves[cwi]].packedCodeword(), *(uint4*)&npc8);
        // Append the score to the region id
        r.append(s, depth);
      } else {
        assert(pNextmoves[cwi] == -1);
      }
    });

    //    thrust::fill(dNextMoves.begin(), dNextMoves.end(), -1);
    dRegionsEnd = thrust::partition(dRegions.begin(), dRegionsEnd,
                                    [] __device__(const RegionID& r) { return !r.isFinalPacked(); });

    printf("%ld regions left\n", dRegionsEnd - dRegions.begin());

    // Sort all games by region id. This re-shuffles within each region only.
    thrust::sort(dRegions.begin(), dRegionsEnd,
                 [] __device__(const RegionID& a, const RegionID& b) { return a.value < b.value; });

    //    for (int i = 0; i < dRegions.size(); i++) std::cout << dRegions[i] << std::endl;
    //    thrust::copy(dRegions.begin(), dRegions.end(), std::ostream_iterator<RegionID>(std::cout, "\n"));

    // Get start and run length for each region by id.
    //   - at the start there are 14, at the end there are 1296
    size_t num_runs =
        thrust::reduce_by_key(dRegions.begin(), dRegionsEnd, thrust::constant_iterator<int>(1),
                              thrust::make_discard_iterator(), runLengths.begin(),
                              [] __device__(const RegionID& a, const RegionID& b) { return a.value == b.value; })
            .second -
        runLengths.begin();

    thrust::exclusive_scan(runLengths.begin(), runLengths.begin() + num_runs, runStarts.begin());

    thrust::sort_by_key(runLengths.begin(), runLengths.begin() + num_runs, runStarts.begin());
    hRunLengths = runLengths;

    // mmmfixme: these could probably be one zipped transform
    thrust::transform(dRegions.begin(), dRegionsEnd, dRegionsAsIndex.begin(),
                      [] __device__(const RegionID& r) { return r.index; });
    thrust::transform(dRegions.begin(), dRegionsEnd, dRegionsAsCodeword.begin(),
                      [pAllCodewords] __device__(const RegionID& r) { return pAllCodewords[r.index]; });

    {
      auto endTime = chrono::high_resolution_clock::now();
      chrono::duration<float, milli> elapsedMS = endTime - startTime;
      auto elapsedS = elapsedMS.count() / 1000;
      cout << "P1 elapsed time " << commaString(elapsedS) << "s" << endl;
      startTime = chrono::high_resolution_clock::now();
    }

    printf("%lu runs\n", num_runs);
    //    for (size_t i = 0; i < num_runs; i++) {
    //      std::cout << runIds[i] << " " << runStarts[i] << "-" << runLengths[i] << endl;
    //    }

    if (num_runs == 0) break;

    // For reach region:
    //   - find next guess using the region as PS, this is the next guess for all games in this region
    //   - games already won (a win at the end of their region id) get no new guess
    //    auto pRunIds = thrust::raw_pointer_cast(runIds.data());
    auto pRunStarts = thrust::raw_pointer_cast(runStarts.data());
    auto pRunLengths = thrust::raw_pointer_cast(runLengths.data());
    auto pRegionsAsIndex = thrust::raw_pointer_cast(dRegionsAsIndex.data());
    auto pRegionsAsCodeword = thrust::raw_pointer_cast(dRegionsAsCodeword.data());

    int ri = 0;
    int regionCnt = 0;
    while (ri < num_runs && hRunLengths[ri] <= 2) {
      regionCnt += hRunLengths[ri];
      ri++;
    }
    printf("Simple runs: %d, totalling %d regions\n", ri, regionCnt);
    if (ri > 0) {
      simpleRuns<<<1, 128>>>(pRegionsAsIndex, pNextmoves, pRunStarts, pRunLengths, ri);
    }

    int bigBoyLaunches = 0;
#if 0
    int pbsIndex = 0;
    for (; ri < num_runs; ri++) {
      auto runStart = runStarts[ri];
      auto runLength = runLengths[ri];

      if (pbsIndex >= chunkSize) {
        pbsIndex = 0;
        //        CubDebug(cudaDeviceSynchronize());
      }
      auto perBlockSolutions = &pdPerBlockSolutions[SubsettingAlgosKernelConfig<SC, uint32_t>::LARGEST_NUM_BLOCKS * pbsIndex++];
      auto littleStuff = &pdLittleStuffs[pbsIndex++];

      bigBoyLaunches++;
      using config8 = SubsettingAlgosKernelConfig<SC, uint8_t>;
      using config16 = SubsettingAlgosKernelConfig<SC, uint16_t>;
      using config32 = SubsettingAlgosKernelConfig<SC, uint32_t>;

      if (config8::shouldUseType(hRunLengths[ri])) {
        subsettingAlgosKernelNew<config8><<<config8::NUM_BLOCKS, config8::THREADS_PER_BLOCK>>>(
            pAllCodewords, pRegionsAsCodeword, runStart, runLength, littleStuff, perBlockSolutions);
        CubDebug(cudaGetLastError());

        // nb: block size on this one must be a power of 2
        reduceMaxScoreNew<128>
            <<<1, 128>>>(perBlockSolutions, config8::NUM_BLOCKS, pRegionsAsIndex, pNextmoves, runStart, runLength);
        CubDebug(cudaGetLastError());
      } else if (config16::shouldUseType(hRunLengths[ri])) {
        subsettingAlgosKernelNew<config16><<<config16::NUM_BLOCKS, config16::THREADS_PER_BLOCK>>>(
            pAllCodewords, pRegionsAsCodeword, runStart, runLength, littleStuff, perBlockSolutions);
        CubDebug(cudaGetLastError());

        // nb: block size on this one must be a power of 2
        reduceMaxScoreNew<128>
            <<<1, 128>>>(perBlockSolutions, config16::NUM_BLOCKS, pRegionsAsIndex, pNextmoves, runStart, runLength);
        CubDebug(cudaGetLastError());
      } else {
        subsettingAlgosKernelNew<config32><<<config32::NUM_BLOCKS, config32::THREADS_PER_BLOCK>>>(
            pAllCodewords, pRegionsAsCodeword, runStart, runLength, littleStuff, perBlockSolutions);
        CubDebug(cudaGetLastError());

        // nb: block size on this one must be a power of 2
        reduceMaxScoreNew<128>
            <<<1, 128>>>(perBlockSolutions, config32::NUM_BLOCKS, pRegionsAsIndex, pNextmoves, runStart, runLength);
        CubDebug(cudaGetLastError());
      }
    }
#else
    for (size_t offset = ri; offset < num_runs; offset += chunkSize) {
      auto runsToDo = min(chunkSize, num_runs - offset);
      int threadsPerBlock = 4;  // mmmfixme: odd...

      bigBoyLaunches++;
      newGuessForRegions<<<(runsToDo + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
          pAllCodewords, pRegionsAsCodeword, pRegionsAsIndex, pNextmoves, pRunStarts, pRunLengths, offset, runsToDo,
          pdPerBlockSolutions);
    }
#endif
    printf("Big Boy Launches: %d\n", bigBoyLaunches);
    CubDebug(cudaDeviceSynchronize());

    {
      auto endTime = chrono::high_resolution_clock::now();
      chrono::duration<float, milli> elapsedMS = endTime - startTime;
      auto elapsedS = elapsedMS.count() / 1000;
      cout << "P2 elapsed time " << commaString(elapsedS) << "s" << endl;
      startTime = chrono::high_resolution_clock::now();
    }

    if (depth > 16) {
      anyNewMoves = false;  // mmmfixme: tmp
    }
  }

  cout << "Last depth: " << depth << endl;

  hRegions = dRegions;
  int maxDepth = 0;
  size_t totalTurns = 0;
  for (int i = 0; i < hRegions.size(); i++) {
    auto c = hRegions[i].countMovesPacked();
    maxDepth = max(maxDepth, c);
    totalTurns += c;
  }

  cout << "Max depth: " << maxDepth << endl;
  printf("Average number of turns was %.4f\n", (double)totalTurns / allCodewords.size());
}

void new_algo::run() {
  vector<CodewordT>& allCodewords = CodewordT ::getAllCodewords();

  // Starting case: all games, initial guess.
  //  vector<CodewordT> used{};
  vector<CodewordT> nextMoves(allCodewords.size(), INITIAL_GUESS);
  vector<RegionID> regions(allCodewords.size());
  for (int i = 0; i < regions.size(); i++) regions[i].index = i;

  int depth = 0;
  size_t maxDepth = 0;
  size_t totalTurns = 0;

  // mmmfixme: turn this into a simple, obviously correct reference impl.
  //  - then do a more optimized version.s

  // If no games have new moves, then we're done
  bool anyNewMoves = true;
  while (anyNewMoves) {
    depth++;
    printf("\n---------- depth = %d ----------\n\n", depth);

    // Score all games against their next guess, if any, which was given per-region
    for (auto& r : regions) {
      auto cwi = r.index;
      if (!nextMoves[cwi].isInvalid()) {
        auto s = allCodewords[cwi].score(nextMoves[cwi]);
        // Append the score to the region id
        r.append(s, depth);
        if (s == CodewordT::WINNING_SCORE) {
          maxDepth = max(maxDepth, (size_t)depth);
          totalTurns += depth;
        }
      }
    }

    // Sort all games by region id. This re-shuffles within each region only.
    std::stable_sort(regions.begin(), regions.end(),
                     [&](const RegionID& a, const RegionID& b) { return a.value < b.value; });

    // Get start and run length for each region by id.
    //   - at the start there are 14, at the end there are 1296
    vector<RegionRun> regionRuns{};
    regionRuns.push_back({regions[0], 0, 0});
    for (int i = 0; i < regions.size(); i++) {
      if (regions[i].value == regionRuns.back().region.value) {
        regionRuns.back().length++;
      } else {
        regionRuns.push_back({regions[i], i, 1});
      }
    }

    printf("%lu regions\n", regionRuns.size());

    // For reach region:
    //   - find next guess using the region as PS, this is the next guess for all games in this region
    //   - games already won (a win at the end of their region id) get no new guess
    anyNewMoves = false;
    for (auto& run : regionRuns) {
      cout << run.region << " -- " << run.start << ":" << run.length << " -- ";
      if (run.region.isFinal()) {
        cout << "win" << endl;
        for (int j = run.start; j < run.start + run.length; j++) {
          nextMoves[regions[j].index] = {};
        }
      } else {
        vector<CodewordT> ps;
        ps.reserve(run.length);
        for (int j = run.start; j < run.start + run.length; j++) {
          ps.push_back(allCodewords[regions[j].index]);
        }
        CodewordT ng = nextGuess(ps, {});  // mmmfixme: what if we need usedCodewords?
        cout << "ng = " << ng << endl;
        for (int j = run.start; j < run.start + run.length; j++) {
          nextMoves[regions[j].index] = ng;
        }
        anyNewMoves = true;
      }
    }
  }

  cout << "Max depth: " << maxDepth << endl;
  printf("Average number of turns was %.4f\n", (double)totalTurns / allCodewords.size());
}
