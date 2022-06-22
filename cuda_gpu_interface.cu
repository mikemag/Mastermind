// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cuda/std/cstdint>
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
// hacks and SWAR action. This is O(1) using warp SIMD intrinsics.
//
// Find black hits with xor, which leaves zero nibbles on matches, then count the zeros in the result. This is a
// variation on determining if a word has a zero byte from https://graphics.stanford.edu/~seander/bithacks.html. This
// part ends with using the GPU's SIMD popcount() to count the zero nibbles.
//
// Next, color counts come from the parallel buffer, and we can run over them and add up total hits, per Knuth[1], by
// aggregating min color counts between the secret and guess.

// TODO: early draft https://godbolt.org/z/ea7YjEPqf

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
  uint allHits = __vsadu4(mins1, 0);
  allHits += __vsadu4(mins2, 0);
  allHits += __vsadu4(mins3, 0);
  allHits += __vsadu4(mins4, 0);

  // Given w = ah - b, simplify to i = bp - ((b - 1)b) / 2) + ah. I wonder if the compiler noticed that.
  // https://godbolt.org/z/ab5vTn -- gcc 10.2 notices and simplifies, clang 11.0.0 misses it.
  return b * pinCount - ((b - 1) * b) / 2 + allHits;
}

// The common portion of the kernels which scores all possible solutions against a given secret and computes subset
// sizes, i.e., for each score the number of codewords with that score.
template <uint32_t pinCount, Algo algo, typename SubsetSize>
__device__ void computeSubsetSizes(SubsetSize *__restrict__ subsetSizes, const uint32_t secret,
                                   const uint4 secretColors, const uint32_t possibleSolutionsCount,
                                   const uint32_t *__restrict__ possibleSolutions,
                                   const uint4 *__restrict__ possibleSolutionsColors) {
  for (uint32_t i = 0; i < possibleSolutionsCount; i++) {
    uint score = scoreCodewords<pinCount>(secret, secretColors, possibleSolutions[i], possibleSolutionsColors[i]);
    if (algo == Algo::MostParts) {
      subsetSizes[score] = 1;
    } else {
      subsetSizes[score]++;
    }
  }
}

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

struct IndexAndScoreReducer {
  __device__ __forceinline__ GPUInterface::IndexAndScore operator()(const GPUInterface::IndexAndScore &a,
                                                                    const GPUInterface::IndexAndScore &b) const {
    // Always take the best score. If it's a tie, take the one that could be a solution. If that's a tie, take lexically
    // first.
    if (b.score > a.score) return b;
    if (b.score < a.score) return a;
    if (b.isPossibleSolution ^ a.isPossibleSolution) return b.isPossibleSolution ? b : a;
    return (b.index < a.index) ? b : a;
  }
};

template <uint32_t pinCount, Algo algo, typename SubsetSize, int totalScores>
__global__ void subsettingAlgosKernel(const uint32_t allCodewordsCount, const uint32_t *__restrict__ allCodewords,
                                      const uint4 *__restrict__ allCodewordsColors, uint32_t possibleSolutionsCount,
                                      const uint32_t *__restrict__ possibleSolutions,
                                      const uint4 *__restrict__ possibleSolutionsColors, uint32_t usedCodewordsCount,
                                      const uint32_t *__restrict__ usedCodewords, uint32_t *__restrict__ fdGuess,
                                      GPUInterface::IndexAndScore *__restrict__ perBlockSolutions) {
  using BlockReduce = cub::BlockReduce<GPUInterface::IndexAndScore, 128>;  // TODO: block size

  union SharedMemLayout {
    SubsetSize scoreCounts;
    BlockReduce::TempStorage reduce;
    GPUInterface::IndexAndScore aggregate;
  };

  extern __shared__ __align__(alignof(SharedMemLayout)) char sharedMem[];

  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  auto subsetSizes = &(reinterpret_cast<SubsetSize *>(sharedMem))[threadIdx.x * totalScores];
  for (int i = 0; i < totalScores; i++) subsetSizes[i] = 0;

  computeSubsetSizes<pinCount, algo>(subsetSizes, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                     possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);

  bool isPossibleSolution = subsetSizes[totalScores - 1] > 0;

  uint32_t largestSubsetSize = 0;
  uint32_t totalUsedSubsets = 0;
  float entropySum = 0.0;
  float expectedSize = 0.0;
  for (int i = 0; i < totalScores; i++) {
    if (subsetSizes[i] > 0) {
      totalUsedSubsets++;
      switch (algo) {
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
  switch (algo) {
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
  if (tidGrid >= allCodewordsCount) score = 0;
  for (int i = 0; i < usedCodewordsCount; i++) {
    if (allCodewords[tidGrid] == usedCodewords[i]) score = 0;
  }

  // Reduce to find the best solution we have in this block. This keeps the codeword index, score, and possible solution
  // indicator together.
  __syncthreads();
  auto &temp_storage = reinterpret_cast<BlockReduce::TempStorage &>(sharedMem);
  GPUInterface::IndexAndScore ias{tidGrid, score, isPossibleSolution};
  GPUInterface::IndexAndScore bestSolution = BlockReduce(temp_storage).Reduce(ias, IndexAndScoreReducer());

  if (threadIdx.x == 0) {
    perBlockSolutions[blockIdx.x] = bestSolution;
  }

  // If we find some guesses which are fully discriminating, we want to pick the first one lexically to play. tidGrid is
  // the same as the ordinal for each member of allCodewords, so we can simply take the min tidGrid.
  if (possibleSolutionsCount <= totalScores) {
    if (totalUsedSubsets == possibleSolutionsCount) {
      atomicMin(fdGuess, tidGrid);
    }
  }
}

// @TODO: try cub::DeviceReduce::Reduce instead.
template <uint32_t blockSize>
__global__ void reduceMaxScore(GPUInterface::IndexAndScore *__restrict__ perBlockSolutions,
                               const uint32_t solutionsCount) {
  uint32_t idx = threadIdx.x;
  IndexAndScoreReducer reduce;
  GPUInterface::IndexAndScore bestScore{0, 0, false};
  for (uint32_t i = idx; i < solutionsCount; i += blockDim.x) {
    bestScore = reduce(bestScore, perBlockSolutions[i]);
  }

  __shared__ GPUInterface::IndexAndScore r[blockSize];
  r[idx] = bestScore;
  __syncthreads();

  for (uint32_t size = blockDim.x / 2; size > 0; size /= 2) {
    if (idx < size) r[idx] = reduce(r[idx], r[idx + size]);
    __syncthreads();
  }

  if (idx == 0) {
    perBlockSolutions[0] = r[0];
  }
}

template <uint8_t p, uint8_t c, Algo a, typename SubsetSize, bool l>
CUDAGPUInterface<p, c, a, SubsetSize, l>::CUDAGPUInterface() {
  dumpDeviceInfo();

  printf("Using GPU: %s\n\n", gpuInfo["GPU Name"].c_str());

  const uint64_t totalCodewords = Codeword<p, c>::totalCodewords;
  threadsPerBlock = std::min(128lu, totalCodewords);
  numBlocks = (totalCodewords + threadsPerBlock - 1) / threadsPerBlock;  // nb: round up!
  roundedTotalCodewords = numBlocks * threadsPerBlock;

  auto block_reduce_temp_bytes = sizeof(typename cub::BlockReduce<int, 128>::TempStorage);
  sharedMemSize = std::max(1 * sizeof(IndexAndScore), block_reduce_temp_bytes);
  sharedMemSize = std::max(sharedMemSize, sizeof(SubsetSize) * totalScores * threadsPerBlock);

  auto mallocManaged = [](auto devPtr, auto size) {
    cudaError_t err = cudaMallocManaged(devPtr, size);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate managed memory (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  };

  mallocManaged((void **)&dAllCodewords, sizeof(*dAllCodewords) * roundedTotalCodewords);
  mallocManaged((void **)&dAllCodewordsColors, sizeof(*dAllCodewordsColors) * roundedTotalCodewords);
  cudaMalloc((void **)&dPossibleSolutions, sizeof(*dPossibleSolutions) * roundedTotalCodewords);
  cudaMalloc((void **)&dPossibleSolutionsColors, sizeof(*dPossibleSolutionsColors) * roundedTotalCodewords);
  dPossibleSolutionsHost = (uint32_t *)malloc(sizeof(*dPossibleSolutionsHost) * roundedTotalCodewords);
  dPossibleSolutionsColorsHost =
      (unsigned __int128 *)malloc(sizeof(*dPossibleSolutionsColorsHost) * roundedTotalCodewords);
  mallocManaged((void **)&dUsedCodewords, sizeof(*dUsedCodewords) * 100);
  mallocManaged((void **)&dFdGuess, sizeof(*dFdGuess) * 1);
  mallocManaged((void **)&dPerBlockSolutions, sizeof(*dPerBlockSolutions) * numBlocks);

  // This is a huge win. The buffer is small, and it gets put into constant memory. 20s -> 8s on MostParts_8p5c.
  cudaMemAdvise(dUsedCodewords, sizeof(*dUsedCodewords) * 100, cudaMemAdviseSetReadMostly, 0);
}

template <uint8_t p, uint8_t c, Algo a, typename SubsetSize, bool l>
void CUDAGPUInterface<p, c, a, SubsetSize, l>::sendComputeCommand() {
  *dFdGuess = Codeword<p, c>::totalCodewords;

  cudaError_t err = cudaSuccess;

  cudaMemcpyAsync(dPossibleSolutions, dPossibleSolutionsHost, sizeof(*dPossibleSolutions) * possibleSolutionsCount,
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dPossibleSolutionsColors, dPossibleSolutionsColorsHost,
                  sizeof(*dPossibleSolutionsColors) * possibleSolutionsCount, cudaMemcpyHostToDevice);

  subsettingAlgosKernel<p, a, SubsetSize, totalScores><<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      Codeword<p, c>::totalCodewords, dAllCodewords, reinterpret_cast<const uint4 *>(dAllCodewordsColors),
      possibleSolutionsCount, dPossibleSolutions, reinterpret_cast<uint4 *>(dPossibleSolutionsColors),
      usedCodewordsCount, dUsedCodewords, dFdGuess, dPerBlockSolutions);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch subsettingAlgosKernel kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  reduceMaxScore<128><<<1, threadsPerBlock>>>(dPerBlockSolutions, numBlocks);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch reduceMaxScore kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to sync device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template <uint8_t p, uint8_t c, Algo a, typename SubsetSize, bool l>
CUDAGPUInterface<p, c, a, SubsetSize, l>::~CUDAGPUInterface() {
  auto freeManaged = [](auto devPtr) {
    cudaError_t err = cudaFree(devPtr);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to free managed memory (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  };

  freeManaged(dAllCodewords);
  freeManaged(dAllCodewordsColors);
  cudaFree(dPossibleSolutions);
  cudaFree(dPossibleSolutionsColors);
  free(dPossibleSolutionsHost);
  free(dPossibleSolutionsColorsHost);
  freeManaged(dUsedCodewords);
  freeManaged(dFdGuess);
  freeManaged(dPerBlockSolutions);
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

template <uint8_t p, uint8_t c, Algo a, typename SubsetSize, bool l>
void CUDAGPUInterface<p, c, a, SubsetSize, l>::dumpDeviceInfo() {
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

// -----------------------------------------------------------------------------------
// Explicit specializations
//
// TODO: I ought to be able to get rid of these, but I need to try to wrangle the
//   conditional build stuff and compiler used for each file all the way up to main to allow
//   the templates to be included everywhere.

#define INST_PCL(p, c, l)                                                 \
  template class CUDAGPUInterface<p, c, Algo::Knuth, uint32_t, l>;        \
  template class CUDAGPUInterface<p, c, Algo::MostParts, uint8_t, l>;     \
  template class CUDAGPUInterface<p, c, Algo::ExpectedSize, uint32_t, l>; \
  template class CUDAGPUInterface<p, c, Algo::Entropy, uint32_t, l>;

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

// INST_L(true)
// INST_L(false)

// INST_PCL(4, 6, true)
// INST_PCL(4, 6, false)
// INST_PCL(8, 5, false)

// The unit test needs this one all the time
template class CUDAGPUInterface<4, 6, Algo::Knuth, uint32_t, true>;

// Specializations for whatever experiments you want to run. Keep this list fairly small to keep compilation speeds
// reasonable. Use the macros above to enable lots of things at once, but with long comp times.
template class CUDAGPUInterface<4, 6, Algo::Knuth, uint32_t, false>;
template class CUDAGPUInterface<8, 5, Algo::Knuth, uint32_t, false>;
template class CUDAGPUInterface<8, 5, Algo::MostParts, uint8_t, false>;
template class CUDAGPUInterface<8, 7, Algo::MostParts, uint8_t, false>;

// Device 0: "NVIDIA GeForce RTX 3070"
//  CUDA Driver Version / Runtime Version          11.7 / 11.7
//  CUDA Capability Major/Minor version number:    8.6
//  Total amount of global memory:                 7949 MBytes (8334737408 bytes)
//  (046) Multiprocessors, (128) CUDA Cores/MP:    5888 CUDA Cores
//  GPU Max Clock rate:                            1755 MHz (1.75 GHz)
//  Memory Clock rate:                             7001 Mhz
//  Memory Bus Width:                              256-bit
//  L2 Cache Size:                                 4194304 bytes
//  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
//  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
//  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
//  Total amount of constant memory:               65536 bytes
//  Total amount of shared memory per block:       49152 bytes
//  Total shared memory per multiprocessor:        102400 bytes
//  Total number of registers available per block: 65536
//  Warp size:                                     32
//  Maximum number of threads per multiprocessor:  1536
//  Maximum number of threads per block:           1024
//  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
//  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
//  Maximum memory pitch:                          2147483647 bytes
//  Texture alignment:                             512 bytes
//  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
//  Run time limit on kernels:                     Yes
//  Integrated GPU sharing Host Memory:            No
//  Support host page-locked memory mapping:       Yes
//  Alignment requirement for Surfaces:            Yes
//  Device has ECC support:                        Disabled
//  Device supports Unified Addressing (UVA):      Yes
//  Device supports Managed Memory:                Yes
//  Device supports Direct Mgd Access From Host:   No
//  Device supports Compute Preemption:            Yes
//  Supports Cooperative Kernel Launch:            Yes
//  Supports MultiDevice Co-op Kernel Launch:      Yes
//  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
//  Compute Mode:
//     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
// Playing all 8 pin 7 color games using algorithm 'Most Parts' for every possible secret...
// Total codewords: 5,764,801
// Initial guess: 11112222
// Completed 11113736, 4.0246s per 1000, 0.02%, ETA 23197.00s
// Completed 11116665, 0.4654s per 1000, 0.03%, ETA 2681.77s
// Completed 11122624, 6.4588s per 1000, 0.05%, ETA 37214.42s
// ...
// Completed 77766666, 0.0017s per 1000, 99.95%, ETA 0.00s
// Completed 77772625, 0.0065s per 1000, 99.97%, ETA 0.01s
// Completed 77775554, 0.0042s per 1000, 99.99%, ETA 0.00s
// Average number of turns was 6.8241
//     Maximum number of turns over all possible secrets was 12 with secret 31624257
//     Elapsed time 1,319.6678s, average search 0.2289ms
//                Codeword comparisons: CPU = 1,174,260,095, GPU = 150,214,203,669,936, total = 150,215,377,930,031
//     GPU kernels executed: 519,935  FPS: 393.99
//
//     1: 1  2: 21  3: 506  4: 11,838  5: 261,471  6: 1,825,202  7: 2,446,268  8: 1,072,842  9: 141,380 10: 5,192 11: 79
//     12: 1
// Done
// Writing all game stats to mastermind_run_stats_20220618_230115_3350c87.csv
