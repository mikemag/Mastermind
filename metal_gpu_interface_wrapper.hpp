// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "gpu_interface.hpp"

#ifdef __OBJC__
#define OBJC_CLASS(name) @class name
#else
#define OBJC_CLASS(name) typedef struct objc_object name
#endif

OBJC_CLASS(MetalGPUInterface);

class MetalGPUInterfaceWrapper : public GPUInterface {
  MetalGPUInterface* wrapped;

 public:
  MetalGPUInterfaceWrapper(unsigned int pinCount, unsigned int totalCodewords, const char* kernelName);
  ~MetalGPUInterfaceWrapper() override;

  bool gpuAvailable() const override { return wrapped != nullptr; }

  uint32_t* getAllCodewordsBuffer() override;
  unsigned __int128* getAllCodewordsColorsBuffer() override;
  void setAllCodewordsCount(uint32_t count) override;
  void syncAllCodewords(uint32_t count) override;

  uint32_t* getPossibleSolutionsBuffer() override;
  unsigned __int128* getPossibleSolutionsColorsBuffer() override;
  void setPossibleSolutionsCount(uint32_t count) override;

  void sendComputeCommand() override;

  uint32_t* getScores() override;
  bool* getRemainingIsPossibleSolution() override;

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override;

  std::string getGPUName() override;
};
