// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "gpu_interface.hpp"

template <typename CodewordT>
class NoGPUInterface : public GPUInterface<CodewordT> {
  unordered_map<string, string> gpuInfo;

 public:
  NoGPUInterface() = default;

  bool gpuAvailable() const override { return false; }

  void sendComputeCommand(const std::vector<CodewordT>& possibleSolutions,
                          const std::vector<uint32_t>& usedCodewords) override {}

  uint32_t getFullyDiscriminatingGuess() override { return UINT32_MAX; }
  IndexAndScore getBestGuess() override { return {UINT32_MAX, 0, false}; }

  std::unordered_map<std::string, std::string>& getGPUInfo() override { return gpuInfo; }
};
