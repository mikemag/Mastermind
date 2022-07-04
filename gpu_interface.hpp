// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "codeword.hpp"

// Keeps an index into the all codewords vector together with a score on the GPU, and whether or not this codeword is a
// possible solution.
struct IndexAndScore {
  uint32_t index;
  uint32_t score;
  bool isPossibleSolution;
  bool isFD;
};

// An interface to keep the GPU code separate from the rest of the code, so it can be compiled in conditionally, or
// support CUDA vs. Metal, etc.
template <typename CodewordT>
class GPUInterface {
 public:
  virtual ~GPUInterface() = default;

  virtual bool gpuAvailable() const = 0;

  virtual void sendComputeCommand(const std::vector<CodewordT>& possibleSolutions,
                                  const std::vector<uint32_t>& usedCodewords) = 0;

  virtual uint32_t getFullyDiscriminatingGuess() = 0;
  virtual IndexAndScore getBestGuess() = 0;

  virtual std::unordered_map<std::string, std::string>& getGPUInfo() = 0;
};
