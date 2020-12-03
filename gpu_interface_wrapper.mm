// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "gpu_interface_wrapper.hpp"
#import "gpu_interface.h"
#include <iostream>
#include <string>

GPUInterfaceWrapper::GPUInterfaceWrapper(unsigned int pinCount, unsigned int totalCodewords, const char *kernelName) {
#ifdef __MM_GPU_METAL__
  NSString *nsKernelName = [[NSString alloc] initWithUTF8String:kernelName];
  wrapped = [[GPUInterface alloc] initWithPinCount:pinCount totalCodewords:totalCodewords kernelName:nsKernelName];
#else
  wrapped = nullptr; // This will indicate there is no GPU available in this build
  std::cout << "Note: GPU not available in this build." << std::endl;
#endif
}

uint32_t *GPUInterfaceWrapper::getAllCodewordsBuffer() {
#ifdef __MM_GPU_METAL__
  return [wrapped getAllCodewordsBuffer];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}

unsigned __int128 *GPUInterfaceWrapper::getAllCodewordsColorsBuffer() {
#ifdef __MM_GPU_METAL__
  return [wrapped getAllCodewordsColorsBuffer];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}

void GPUInterfaceWrapper::setAllCodewordsCount(uint32_t count) {
#ifdef __MM_GPU_METAL__
  [wrapped setAllCodewordsCount:count];
#else
  assert(!"No GPU support!");
#endif
}

void GPUInterfaceWrapper::syncAllCodewords(uint32_t count) {
#ifdef __MM_GPU_METAL__
  [wrapped syncAllCodewords:count];
#else
  assert(!"No GPU support!");
#endif
}

uint32_t *GPUInterfaceWrapper::getPossibleSolutionssBuffer() {
#ifdef __MM_GPU_METAL__
  return [wrapped getPossibleSolutionssBuffer];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}

unsigned __int128 *GPUInterfaceWrapper::getPossibleSolutionsColorsBuffer() {
#ifdef __MM_GPU_METAL__
  return [wrapped getPossibleSolutionsColorsBuffer];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}

void GPUInterfaceWrapper::setPossibleSolutionsCount(uint32_t count) {
#ifdef __MM_GPU_METAL__
  return [wrapped setPossibleSolutionsCount:count];
#else
  assert(!"No GPU support!");
#endif
}

void GPUInterfaceWrapper::sendComputeCommand() {
#ifdef __MM_GPU_METAL__
  [wrapped sendComputeCommand];
#else
  assert(!"No GPU support!");
#endif
}

uint32_t *GPUInterfaceWrapper::getScores() {
#ifdef __MM_GPU_METAL__
  return [wrapped getScores];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}

bool *GPUInterfaceWrapper::getRemainingIsPossibleSolution() {
#ifdef __MM_GPU_METAL__
  return [wrapped getRemainingIsPossibleSolution];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}

uint32_t *GPUInterfaceWrapper::getFullyDiscriminatingCodewords(uint32_t &count) {
#ifdef __MM_GPU_METAL__
  return [wrapped getFullyDiscriminatingCodewords:&count];
#else
  assert(!"No GPU support!");
  return nullptr;
#endif
}


std::string GPUInterfaceWrapper::getGPUName() {
#ifdef __MM_GPU_METAL__
  return [wrapped getGPUName].UTF8String;
#else
  assert(!"No GPU support!");
  return "";
#endif
}
