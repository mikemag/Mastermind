// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#ifdef __OBJC__
#define OBJC_CLASS(name) @class name
#else
#define OBJC_CLASS(name) typedef struct objc_object name
#endif

OBJC_CLASS(GPUInterface);

class GPUInterfaceWrapper {
  GPUInterface* wrapped;

 public:
  GPUInterfaceWrapper(unsigned int pinCount, unsigned int totalCodewords, const char* kernelName);

  bool gpuAvailable() const { return wrapped != nullptr; }

  uint32_t* getAllCodewordsBuffer();
  unsigned __int128* getAllCodewordsColorsBuffer();
  void setAllCodewordsCount(uint32_t count);
  void syncAllCodewords(uint32_t count);

  uint32_t* getPossibleSolutionssBuffer();
  unsigned __int128* getPossibleSolutionsColorsBuffer();
  void setPossibleSolutionsCount(uint32_t count);

  void sendComputeCommand();

  uint32_t* getScores();
  bool* getRemainingIsPossibleSolution();

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count);

  std::string getGPUName();
};
