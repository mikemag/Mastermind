// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda_runtime.h>

#include <algorithm>
#define CUB_STDERR
#include <cub/cub.cuh>
#include <cuda/std/cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "codeword.hpp"
#include "compute_kernel_constants.h"
#include "utils.hpp"

using namespace std;

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

// The common portion of the kernels which scores all possible solutions against a given secret and computes subset
// sizes, i.e., for each score the number of codewords with that score.
template <uint32_t PIN_COUNT, Algo ALGO, typename SubsetSizeT, typename CodewordT>
__device__ void computeSubsetSizes(SubsetSizeT *__restrict__ subsetSizes, const uint32_t secret,
                                   const uint4 secretColors, const uint32_t possibleSolutionsCount,
                                   const CodewordT *__restrict__ possibleSolutions) {
  for (uint32_t i = 0; i < possibleSolutionsCount; i++) {
    unsigned __int128 pc8 = possibleSolutions[i].packedColors8();  // Annoying...
    uint score = scoreCodewords<PIN_COUNT>(secret, secretColors, possibleSolutions[i].packedCodeword(), *(uint4 *)&pc8);
    if (ALGO == Algo::MostParts) {
      subsetSizes[score] = 1;
    } else {
      subsetSizes[score]++;
    }
  }
}

// Holds all the constants we need to kick off a CUDA kernel for all the subsetting strategies given a strategy config.
// Computes how many threads per block, blocks needed, and importantly shared memory size. Can override the subset
// counter type to be smaller than the one given by the Strategy when we know the max subset size is small enough.
template <typename SubsettingStrategyConfig, typename SubsetSizeOverrideT = uint32_t>
struct SubsettingAlgosKernelConfig {
  static constexpr uint8_t PIN_COUNT = SubsettingStrategyConfig::PIN_COUNT;
  static constexpr uint8_t COLOR_COUNT = SubsettingStrategyConfig::COLOR_COUNT;
  static constexpr bool LOG = SubsettingStrategyConfig::LOG;
  static constexpr Algo ALGO = SubsettingStrategyConfig::ALGO;
  using CodewordT = Codeword<PIN_COUNT, COLOR_COUNT>;

  // Total scores = (PIN_COUNT * (PIN_COUNT + 3)) / 2, but +1 for imperfect packing.
  static constexpr int TOTAL_PACKED_SCORES = ((PIN_COUNT * (PIN_COUNT + 3)) / 2) + 1;

  using SubsetSizeT =
      typename std::conditional<sizeof(SubsetSizeOverrideT) < sizeof(typename SubsettingStrategyConfig::SubsetSizeT),
                                SubsetSizeOverrideT, typename SubsettingStrategyConfig::SubsetSizeT>::type;

  // This subset size is good given the PS size, or this is the default type provided by the Strategy.
  // This is quite coarse: sure, no subset can be larger that PS size, but it's also unlikely that any PS has a single
  // subset. Provable? If so, with a better bound?
  static bool shouldUseType(uint32_t possibleSolutionsCount) {
    return possibleSolutionsCount < numeric_limits<SubsetSizeT>::max() ||
           sizeof(SubsetSizeOverrideT) == sizeof(typename SubsettingStrategyConfig::SubsetSizeT);
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

  using BlockReduce = cub::BlockReduce<IndexAndScore, THREADS_PER_BLOCK>;

  union SharedMemLayout {
    SubsetSizeT subsetSizes[TOTAL_PACKED_SCORES * THREADS_PER_BLOCK];
    typename BlockReduce::TempStorage reducerTmpStorage;
    IndexAndScore aggregate;  // Ensure alignment for these
  };

  // Confirm the shared mem size is as expected
  static_assert(sizeof(SharedMemLayout) ==
                std::max({
                    sizeof(SubsetSizeT) * TOTAL_PACKED_SCORES * THREADS_PER_BLOCK,
                    sizeof(typename cub::BlockReduce<IndexAndScore, THREADS_PER_BLOCK>::TempStorage),
                    1 * sizeof(IndexAndScore),
                }));
};

// Little tests
using testConfig = SubsettingAlgosKernelConfig<SubsettingStrategyConfig<8, 5, false, Algo::Knuth, uint32_t>>;
static_assert(nextPowerOfTwo(uint32_t(136)) == 256);
static_assert(testConfig::maxThreadsFromSubsetType<uint32_t>() == 256);
static_assert(testConfig::numBlocks(testConfig::threadsPerBlock<uint32_t>()) == 1526);

// Reducer for per-thread scores, used for CUB per-block and device reductions.
struct IndexAndScoreReducer {
  __device__ __forceinline__ IndexAndScore operator()(const IndexAndScore &a, const IndexAndScore &b) const {
    // Always take the best score. If it's a tie, take the one that could be a solution. If that's a tie, take lexically
    // first.
    if (b.score > a.score) return b;
    if (b.score < a.score) return a;
    if (b.isPossibleSolution ^ a.isPossibleSolution) return b.isPossibleSolution ? b : a;
    return (b.index < a.index) ? b : a;
  }
};

// This takes two sets of codewords: the "all codewords" set, which is every possible codeword, and the "possible
// solutions" set. They're separated into two buffers each: one for codewords, which are packed into 32 bits, and
// one for pre-computed color counts, packed into 128 bits as 16 8-bit counters.
//
// The all codewords set is placed into GPU memory once at program start and remains constant.
//
// The possible solutions set changes each time, both content and length, but reuses the same buffers.
//
// All codeword pairs are scored and subset sizes computed, then each codeword is scored for the algorithm we're
// running. Finally, each block computes the best scored codeword in the group, and we look for fully discriminating
// codewords.
//
// Output is an array of IndexAndScores for the best selections from each block, and a single fully discriminating
// guess.
//
// Finally, there's shared block memory for each thread with enough room for all the intermediate subset sizes,
// reduction space, etc.
template <typename SubsettingAlgosKernelConfig, typename LittleStuffT>
__global__ void subsettingAlgosKernel(
    const uint32_t *__restrict__ allCodewords, const uint4 *__restrict__ allCodewordsColors,
    uint32_t possibleSolutionsCount,
    const typename SubsettingAlgosKernelConfig::CodewordT *__restrict__ possibleSolutions,
    LittleStuffT *__restrict__ littleStuff, IndexAndScore *__restrict__ perBlockSolutions) {
  __shared__ typename SubsettingAlgosKernelConfig::SharedMemLayout sharedMem;

  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  auto subsetSizes = &sharedMem.subsetSizes[threadIdx.x * SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES];
  for (int i = 0; i < SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES; i++) subsetSizes[i] = 0;

  computeSubsetSizes<SubsettingAlgosKernelConfig::PIN_COUNT, SubsettingAlgosKernelConfig::ALGO>(
      subsetSizes, allCodewords[tidGrid], allCodewordsColors[tidGrid], possibleSolutionsCount, possibleSolutions);

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
  if (tidGrid >= SubsettingAlgosKernelConfig::CodewordT::TOTAL_CODEWORDS) score = 0;
  for (int i = 0; i < littleStuff->usedCodewordsCount; i++) {
    if (allCodewords[tidGrid] == littleStuff->usedCodewords[i]) score = 0;
  }

  // Reduce to find the best solution we have in this block. This keeps the codeword index, score, and possible solution
  // indicator together.
  __syncthreads();
  IndexAndScore ias{tidGrid, score, isPossibleSolution};
  IndexAndScore bestSolution = typename SubsettingAlgosKernelConfig::BlockReduce(sharedMem.reducerTmpStorage)
                                   .Reduce(ias, IndexAndScoreReducer());

  if (threadIdx.x == 0) {
    perBlockSolutions[blockIdx.x] = bestSolution;
  }

  // If we find some guesses which are fully discriminating, we want to pick the first one lexically to play. tidGrid is
  // the same as the ordinal for each member of allCodewords, so we can simply take the min tidGrid.
  if (possibleSolutionsCount <= SubsettingAlgosKernelConfig::TOTAL_PACKED_SCORES) {
    if (totalUsedSubsets == possibleSolutionsCount) {
      // I don't really like this, but it's tested out faster than doing a per-block reduction and a subsequent
      // device-wide reduction, like for the index and score above. Likewise, doing a warp-level reduction centered
      // around __reduce_min_sync() tests the same speed as just the atomicMin().
      atomicMin(&(littleStuff->fdGuess), tidGrid);
    }
  }
}

// cub::DeviceReduce::Reduce is a slight loss to this, not sure why, so keeping the custom kernel for now.
template <uint32_t blockSize, typename LittleStuffT>
__global__ void reduceMaxScore(IndexAndScore *__restrict__ perBlockSolutions, const uint32_t solutionsCount,
                               LittleStuffT *__restrict__ littleStuff) {
  uint32_t idx = threadIdx.x;
  IndexAndScoreReducer reduce;
  IndexAndScore bestScore{0, 0, false};
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

  if (idx == 0) {
    littleStuff->bestGuess = shared[0];
  }
}

template <typename SubsettingStrategyConfig>
CUDAGPUInterface<SubsettingStrategyConfig>::CUDAGPUInterface(
    const std::vector<typename SubsettingStrategyConfig::CodewordT> &allCodewords) {
  dumpDeviceInfo();

  printf("Using GPU: %s\n\n", gpuInfo["GPU Name"].c_str());

  // nb: the subset size type doesn't matter for this config since we only use the largest values and nothing else.
  using config = SubsettingAlgosKernelConfig<SubsettingStrategyConfig>;
  auto roundedTotalCodewords = config::LARGEST_ROUNDED_TOTAL_CODEWORDS;
  auto numBlocks = config::LARGEST_NUM_BLOCKS;

  CubDebugExit(cudaMallocManaged((void **)&dAllCodewords, sizeof(*dAllCodewords) * roundedTotalCodewords));
  CubDebugExit(cudaMallocManaged((void **)&dAllCodewordsColors, sizeof(*dAllCodewordsColors) * roundedTotalCodewords));
  CubDebugExit(cudaMalloc((void **)&dPossibleSolutions, sizeof(*dPossibleSolutions) * roundedTotalCodewords));
  CubDebugExit(cudaMalloc((void **)&dLittleStuff, sizeof(*dLittleStuff)));
  CubDebugExit(cudaMalloc((void **)&dPerBlockSolutions, sizeof(*dPerBlockSolutions) * numBlocks));

  // Setup AC set once
  uint32_t *acw = dAllCodewords;
  unsigned __int128 *acc = dAllCodewordsColors;
  for (int i = 0; i < allCodewords.size(); i++) {
    acw[i] = allCodewords[i].packedCodeword();
    acc[i] = allCodewords[i].packedColors8();
  }

  CubDebugExit(
      cudaMemAdvise(dAllCodewords, sizeof(*dAllCodewords) * roundedTotalCodewords, cudaMemAdviseSetReadMostly, 0));
  CubDebugExit(cudaMemAdvise(dAllCodewordsColors, sizeof(*dAllCodewordsColors) * roundedTotalCodewords,
                             cudaMemAdviseSetReadMostly, 0));
}

template <typename SubsettingStrategyConfig>
template <typename SubsettingAlgosKernelConfig>
void CUDAGPUInterface<SubsettingStrategyConfig>::launchSubsettingKernel(uint32_t possibleSolutionsCount) {
  subsettingAlgosKernel<SubsettingAlgosKernelConfig>
      <<<SubsettingAlgosKernelConfig::NUM_BLOCKS, SubsettingAlgosKernelConfig::THREADS_PER_BLOCK>>>(
          dAllCodewords, reinterpret_cast<const uint4 *>(dAllCodewordsColors), possibleSolutionsCount,
          dPossibleSolutions, dLittleStuff, dPerBlockSolutions);
  CubDebugExit(cudaGetLastError());

  // nb: block size on this one must be a power of 2
  reduceMaxScore<128><<<1, 128>>>(dPerBlockSolutions, SubsettingAlgosKernelConfig::NUM_BLOCKS, dLittleStuff);
  CubDebugExit(cudaGetLastError());

  // Bring back results from both kernels: best guess from reduceMaxScore, and the FD guess from subsettingAlgosKernel.
  CubDebugExit(cudaMemcpyAsync(&littleStuff, dLittleStuff, sizeof(littleStuff), cudaMemcpyDeviceToHost));

  CubDebugExit(cudaDeviceSynchronize());
}

template <typename SubsettingStrategyConfig>
void CUDAGPUInterface<SubsettingStrategyConfig>::sendComputeCommand(
    const std::vector<typename SubsettingStrategyConfig::CodewordT> &possibleSolutions,
    const std::vector<uint32_t> &usedCodewords) {
  auto psSize = possibleSolutions.size();
  CubDebugExit(cudaMemcpyAsync(dPossibleSolutions, possibleSolutions.data(), sizeof(*dPossibleSolutions) * psSize,
                               cudaMemcpyHostToDevice));

  littleStuff.fdGuess =
      Codeword<SubsettingStrategyConfig::PIN_COUNT, SubsettingStrategyConfig::COLOR_COUNT>::TOTAL_CODEWORDS;
  memcpy(littleStuff.usedCodewords, usedCodewords.data(), usedCodewords.size() * sizeof(*littleStuff.usedCodewords));
  littleStuff.usedCodewordsCount = usedCodewords.size();
  CubDebugExit(cudaMemcpyAsync(dLittleStuff, &littleStuff, sizeof(littleStuff), cudaMemcpyHostToDevice));

  using config8 = SubsettingAlgosKernelConfig<SubsettingStrategyConfig, uint8_t>;
  using config16 = SubsettingAlgosKernelConfig<SubsettingStrategyConfig, uint16_t>;
  using config32 = SubsettingAlgosKernelConfig<SubsettingStrategyConfig, uint32_t>;

  totalSubsettingKernels++;
  if (config8::shouldUseType(psSize)) {
    psSizesIn8Bits++;
    launchSubsettingKernel<config8>(psSize);
  } else if (config16::shouldUseType(psSize)) {
    psSizesIn16Bits++;
    launchSubsettingKernel<config16>(psSize);
  } else {
    psSizesIn32Bits++;
    launchSubsettingKernel<config32>(psSize);
  }
}

template <typename SubsettingStrategyConfig>
CUDAGPUInterface<SubsettingStrategyConfig>::~CUDAGPUInterface() {
  CubDebugExit(cudaFree(dAllCodewords));
  CubDebugExit(cudaFree(dAllCodewordsColors));
  CubDebugExit(cudaFree(dPossibleSolutions));
  CubDebugExit(cudaFree(dLittleStuff));
  CubDebugExit(cudaFree(dPerBlockSolutions));
}

// ------------------------------------------------------------------------------------------------------------
// Dumping device info from https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128},
                                     {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},
                                     {0x75, 64},  {0x80, 64},  {0x86, 128}, {0x87, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor,
         nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

template <typename SubsettingStrategyConfig>
void CUDAGPUInterface<SubsettingStrategyConfig>::dumpDeviceInfo() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }

  int dev = 0;  // TODO: handle multiple GPU's and select the best one.
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  printf("Device %d: \"%s\"\n", dev, deviceProp.name);
  gpuInfo["GPU Name"] = string(deviceProp.name);

  int driverVersion = 0, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  gpuInfo["GPU CUDA Driver Version"] = to_string(driverVersion / 1000) + "." + to_string((driverVersion % 100) / 10);
  gpuInfo["GPU CUDA Runtime Version"] = to_string(runtimeVersion / 1000) + "." + to_string((runtimeVersion % 100) / 10);

  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
  gpuInfo["GPU CUDA Capability"] = to_string(deviceProp.major) + "." + to_string(deviceProp.minor);

  char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(msg, sizeof(msg),
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f), (unsigned long long)deviceProp.totalGlobalMem);
#else
  snprintf(msg, sizeof(msg), "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
           static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f), (unsigned long long)deviceProp.totalGlobalMem);
  gpuInfo["GPU Global Memory"] = to_string(deviceProp.totalGlobalMem);
#endif
  printf("%s", msg);

  printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n", deviceProp.multiProcessorCount,
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
  gpuInfo["GPU Multiprocessors"] = to_string(deviceProp.multiProcessorCount);
  gpuInfo["GPU Cores/MP"] = to_string(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
  gpuInfo["GPU Cores"] =
      to_string(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

  printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f,
         deviceProp.clockRate * 1e-6f);
  gpuInfo["GPU Max Clock Rate MHz"] = to_string(deviceProp.clockRate * 1e-3f);

#if CUDART_VERSION >= 5000
  // This is supported in CUDA 5.0 (runtime API device properties)
  printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);
  gpuInfo["GPU Memory Clock Rate MHz"] = to_string(deviceProp.memoryClockRate * 1e-3f);
  gpuInfo["GPU Memory Bus Width"] = to_string(deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
  }
  gpuInfo["GPU L2 Cache Size"] = to_string(deviceProp.l2CacheSize);

#else
  // This only available in CUDA 4.0-4.2 (but these were only exposed in the
  // CUDA Driver API)
  int memoryClock;
  getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
  printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
  int memBusWidth;
  getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
  printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
  int L2CacheSize;
  getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

  if (L2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
  }
#endif

  printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
         deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
         deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n", deviceProp.maxTexture1DLayered[0],
         deviceProp.maxTexture1DLayered[1]);
  printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n", deviceProp.maxTexture2DLayered[0],
         deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

  printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
  gpuInfo["GPU Constant Memory"] = to_string(deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
  gpuInfo["GPU Shared Memory per block"] = to_string(deviceProp.sharedMemPerBlock);
  printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
  gpuInfo["GPU Shared Memory per MP"] = to_string(deviceProp.sharedMemPerMultiprocessor);
  printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  gpuInfo["GPU Registers per block"] = to_string(deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n", deviceProp.warpSize);
  gpuInfo["GPU Warp Size"] = to_string(deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
  gpuInfo["GPU Threads per MP"] = to_string(deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
  gpuInfo["GPU Threads per Block"] = to_string(deviceProp.maxThreadsPerBlock);
  printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
         deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0],
         deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch);
  printf("  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment);
  printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
         (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
  printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
  printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
  printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
  printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
  printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
         deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
  printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
  printf("  Device supports Managed Memory:                %s\n", deviceProp.managedMemory ? "Yes" : "No");
  printf("  Device supports Direct Mgd Access From Host:   %s\n",
         deviceProp.directManagedMemAccessFromHost ? "Yes" : "No");
  printf("  Device supports Compute Preemption:            %s\n", deviceProp.computePreemptionSupported ? "Yes" : "No");
  printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
  printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
         deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
  printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID,
         deviceProp.pciDeviceID);

  const char *sComputeMode[] = {
      "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
      "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
      "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
      "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
      "Unknown",
      NULL};
  printf("  Compute Mode:\n");
  printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

  printf("\n");
}
