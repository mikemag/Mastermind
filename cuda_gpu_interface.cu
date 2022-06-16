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

// The common portion of the kernels which scores all possible solutions against a given guess and computes subset
// sizes.
typedef uint32_t ScoreCounter;

template <uint32_t pinCount, Algo algo>
__device__ void computeSubsetSizes(ScoreCounter *__restrict__ scoreCounts, const uint32_t secret,
                                   const uint4 secretColors, const uint32_t possibleSolutionsCount,
                                   const uint32_t *__restrict__ possibleSolutions,
                                   const uint4 *__restrict__ possibleSolutionsColors) {
  for (uint32_t i = 0; i < possibleSolutionsCount; i++) {
    uint score = scoreCodewords<pinCount>(secret, secretColors, possibleSolutions[i], possibleSolutionsColors[i]);
    if (algo == Algo::MostParts) {
      scoreCounts[score] = 1;
    } else {
      scoreCounts[score]++;
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
// Output is two arrays with elements for each of the all codewords set: one with the score, and one bool
// for whether the codeword is a possible solution.
//
// Finally, there's shared block memory for each thread with enough room for all the intermediate subset sizes.

extern __shared__ ScoreCounter tgScoreCounts[];

template <uint32_t pinCount, Algo algo, int totalScores>
__global__ void subsettingAlgosKernel(const uint32_t *__restrict__ allCodewords,
                                      const uint4 *__restrict__ allCodewordsColors, uint32_t possibleSolutionsCount,
                                      const uint32_t *__restrict__ possibleSolutions,
                                      const uint4 *__restrict__ possibleSolutionsColors, uint32_t *__restrict__ scores,
                                      bool *__restrict__ remainingIsPossibleSolution,
                                      uint32_t *__restrict__ fullyDiscriminatingCodewords) {
  uint tidGrid = blockDim.x * blockIdx.x + threadIdx.x;
  ScoreCounter *scoreCounts = &tgScoreCounts[threadIdx.x * totalScores];
  for (int i = 0; i < totalScores; i++) scoreCounts[i] = 0;

  computeSubsetSizes<pinCount, algo>(scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                     possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);

  remainingIsPossibleSolution[tidGrid] = scoreCounts[totalScores - 1] > 0;

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
  dumpDeviceInfo();

  const uint64_t totalCodewords = Codeword<p, c>::totalCodewords;
  threadsPerBlock = std::min(128lu, totalCodewords);
  numBlocks = (totalCodewords + threadsPerBlock - 1) / threadsPerBlock;  // nb: round up!
  uint32_t roundedTotalCodewords = numBlocks * threadsPerBlock;
  sharedMemSize = sizeof(ScoreCounter) * totalScores * threadsPerBlock;

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

  subsettingAlgosKernel<p, a, totalScores><<<numBlocks, threadsPerBlock, sharedMemSize>>>(
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
  return deviceName;
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

template <uint8_t p, uint8_t c, Algo a, bool l>
void CUDAGPUInterface<p, c, a, l>::dumpDeviceInfo() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }

  for (int dev = 0; dev < nDevices; dev++) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    deviceName = string(deviceProp.name);

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
              "  Total amount of global memory:                 %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
              (unsigned long long)deviceProp.totalGlobalMem);
#else
    snprintf(msg, sizeof(msg), "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f), (unsigned long long)deviceProp.totalGlobalMem);
#endif
    printf("%s", msg);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n", deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    }

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
    printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
           deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

    printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
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
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID,
           deviceProp.pciBusID, deviceProp.pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
        NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
  }
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
