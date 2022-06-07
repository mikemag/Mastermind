// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

class GPUInterface {
 public:
  virtual ~GPUInterface() = default;

  virtual bool gpuAvailable() const = 0;

  virtual uint32_t* getAllCodewordsBuffer() = 0;
  virtual unsigned __int128* getAllCodewordsColorsBuffer() = 0;
  virtual void setAllCodewordsCount(uint32_t count) = 0;
  virtual void syncAllCodewords(uint32_t count) = 0;

  virtual uint32_t* getPossibleSolutionsBuffer() = 0;
  virtual unsigned __int128* getPossibleSolutionsColorsBuffer() = 0;
  virtual void setPossibleSolutionsCount(uint32_t count) = 0;

  virtual void sendComputeCommand() = 0;

  virtual uint32_t* getScores() = 0;
  virtual bool* getRemainingIsPossibleSolution() = 0;

  virtual uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) = 0;

  virtual std::string getGPUName() = 0;
};
