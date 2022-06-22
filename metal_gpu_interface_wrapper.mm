// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "metal_gpu_interface_wrapper.hpp"
#include <cassert>
#include <iostream>
#include <string>

#import "metal_gpu_interface.h"

MetalGPUInterfaceWrapper::MetalGPUInterfaceWrapper(unsigned int pinCount, unsigned int totalCodewords, const char *kernelName) {
  NSString *nsKernelName = [[NSString alloc] initWithUTF8String:kernelName];
  wrapped = [[MetalGPUInterface alloc] initWithPinCount:pinCount totalCodewords:totalCodewords kernelName:nsKernelName];
}

MetalGPUInterfaceWrapper::~MetalGPUInterfaceWrapper() {
  wrapped = nil;
}

uint32_t *MetalGPUInterfaceWrapper::getAllCodewordsBuffer() {
  return [wrapped getAllCodewordsBuffer];
}

unsigned __int128 *MetalGPUInterfaceWrapper::getAllCodewordsColorsBuffer() {
  return [wrapped getAllCodewordsColorsBuffer];
}

void MetalGPUInterfaceWrapper::setAllCodewordsCount(uint32_t count) {
  [wrapped setAllCodewordsCount:count];
}

void MetalGPUInterfaceWrapper::syncAllCodewords(uint32_t count) {
  [wrapped syncAllCodewords:count];
}

uint32_t *MetalGPUInterfaceWrapper::getPossibleSolutionsBuffer() {
  return [wrapped getPossibleSolutionssBuffer];
}

unsigned __int128 *MetalGPUInterfaceWrapper::getPossibleSolutionsColorsBuffer() {
  return [wrapped getPossibleSolutionsColorsBuffer];
}

void MetalGPUInterfaceWrapper::setPossibleSolutionsCount(uint32_t count) {
  return [wrapped setPossibleSolutionsCount:count];
}

void MetalGPUInterfaceWrapper::sendComputeCommand() {
  [wrapped sendComputeCommand];
}

uint32_t *MetalGPUInterfaceWrapper::getScores() {
  return [wrapped getScores];
}

bool *MetalGPUInterfaceWrapper::getRemainingIsPossibleSolution() {
  return [wrapped getRemainingIsPossibleSolution];
}

uint32_t *MetalGPUInterfaceWrapper::getFullyDiscriminatingCodewords(uint32_t &count) {
  return [wrapped getFullyDiscriminatingCodewords:&count];
}

std::unordered_map<std::string, std::string> &MetalGPUInterfaceWrapper::getGPUInfo() {
  [[wrapped getGPUInfo] enumerateKeysAndObjectsUsingBlock:^(NSString* key, NSString* value, BOOL* stop) {
    gpuInfo[key.UTF8String] = value.UTF8String;
  }];
  return gpuInfo;
}
