// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

using namespace std;

// --------------------------------------------------------------------------------------------------------------------
// Base for all subsetting strategies

template <typename StrategyConfig, class Derived>
typename Strategy<StrategyConfig>::CodewordT StrategySubsetting<StrategyConfig, Derived>::selectNextGuess() {
  // Add the last guess from the list of used codewords.
  usedCodewords = savedUsedCodewords;
  usedCodewords.emplace_back(this->guess.packedCodeword());

  CodewordT bestGuess;
  size_t bestScore = 0;
  bool bestIsPossibleSolution = false;
  int allCount = 0;  // For metrics only
  for (const auto &g : CodewordT::getAllCodewords()) {
    bool isPossibleSolution = false;
    for (const auto &ps : this->possibleSolutions) {
      Score r = g.score(ps);
      ssHolder->subsetSizes[r.result]++;
      if (r == CodewordT::WINNING_SCORE) {
        isPossibleSolution = true;  // Remember if this guess is in the set of possible solutions
      }
    }
    this->rootData->scoreCounterCPU += this->possibleSolutions.size();
    if (this->rootData->ENABLE_SMALL_PS_METRICS) allCount++;

    // Shortcut small sets. Note, we've already done this check for possible solutions.
    if (Strategy<StrategyConfig>::ENABLE_SMALL_PS_SHORTCUT && !isPossibleSolution &&
        this->possibleSolutions.size() <= StrategyConfig::TOTAL_SCORES) {
      int subsetsCount = computeTotalSubsets();
      if (subsetsCount == this->possibleSolutions.size()) {
        if (StrategyConfig::LOG) {
          cout << "Selecting fully discriminating guess: " << g << ", subsets: " << subsetsCount << endl;
        }
        // Done implicitly in computeSubsetScore, which we're skipping
        fill(begin(ssHolder->subsetSizes), end(ssHolder->subsetSizes), 0);
        ++this->rootData->smallPSInnerShortcuts;
        this->rootData->smallPSInnerScoresSkipped +=
            (CodewordT::getAllCodewords().size() - allCount) * this->possibleSolutions.size();
        return g;
      }
      ++this->rootData->smallPSInnerWasted;
    }

    size_t score = computeSubsetScore();

    if (score > bestScore || (!bestIsPossibleSolution && isPossibleSolution && score == bestScore)) {
      if (find(usedCodewords.cbegin(), usedCodewords.cend(), g.packedCodeword()) != usedCodewords.end()) {
        continue;  // Ignore codewords we've already used
      }
      bestScore = score;
      bestGuess = g;
      bestIsPossibleSolution = isPossibleSolution;
    }
  }

  if (StrategyConfig::LOG) {
    cout << "Selecting best guess: " << bestGuess << "\tscore: " << bestScore << endl;
  }
  return bestGuess;
}

template <typename StrategyConfig, class Derived>
int StrategySubsetting<StrategyConfig, Derived>::computeTotalSubsets() {
  int totalSubsets = 0;
  for (auto s : ssHolder->subsetSizes) {
    if (s > 0) {
      totalSubsets++;
    }
  }
  return totalSubsets;
}

// --------------------------------------------------------------------------------------------------------------------
// Base for all subsetting GPU strategies

template <typename StrategyConfig, class Derived>
typename Strategy<StrategyConfig>::CodewordT StrategySubsettingGPU<StrategyConfig, Derived>::selectNextGuess() {
  auto &allCodewords = CodewordT::getAllCodewords();

  // Cut off "small" work and just do it on the CPU.
  if (mode == CPU || (mode == Both && allCodewords.size() * this->possibleSolutions.size() < 4 * 1024)) {
    return StrategySubsetting<StrategyConfig, Derived>::selectNextGuess();
  }

  // Add the last guess from the list of used codewords.
  this->usedCodewords = this->savedUsedCodewords;
  this->usedCodewords.emplace_back(this->guess.packedCodeword());

  gpuRootData->gpuInterface->sendComputeCommand(this->possibleSolutions, this->usedCodewords);
  gpuRootData->kernelsExecuted++;
  this->rootData->scoreCounterGPU += allCodewords.size() * this->possibleSolutions.size();

  if (Strategy<StrategyConfig>::ENABLE_SMALL_PS_SHORTCUT_GPU &&
      this->possibleSolutions.size() <= StrategyConfig::TOTAL_SCORES) {
    // Shortcut small sets with fully discriminating codewords. Certain versions of the GPU kernels look for these and
    // pass a fully descriminating guess back.
    uint32_t fd = gpuRootData->gpuInterface->getFullyDiscriminatingGuess();
    if (fd < CodewordT::TOTAL_CODEWORDS) {
      CodewordT g = allCodewords[fd];
      if (StrategyConfig::LOG) {
        cout << "Selecting fully discriminating guess from GPU: " << g << ", subsets: "
             << this->possibleSolutions.size()
             //             << ", isPossibleSolution: " << remainingIsPossibleSolution[fd]
             << ", small opts index: " << fd << endl;
      }
      return g;
    }
  }

  IndexAndScore bestGPUGuess = gpuRootData->gpuInterface->getBestGuess();
  CodewordT bestGuess = allCodewords[bestGPUGuess.index];
  if (StrategyConfig::LOG) {
    cout << "Selecting best guess: " << bestGuess << "\tscore: " << bestGPUGuess.score << " (GPU)" << endl;
  }

  return bestGuess;
}

template <typename StrategyConfig, class Derived>
void StrategySubsettingGPU<StrategyConfig, Derived>::printStats(chrono::duration<float, milli> elapsedMS) {
  StrategySubsetting<StrategyConfig, Derived>::printStats(elapsedMS);
  if (mode != CPU && gpuRootData->gpuInterface->gpuAvailable()) {
    cout << "GPU kernels executed: " << commaString(gpuRootData->kernelsExecuted)
         << "  FPS: " << commaString((float)gpuRootData->kernelsExecuted / (elapsedMS.count() / 1000.0)) << endl;
  }
}

template <typename StrategyConfig, class Derived>
void StrategySubsettingGPU<StrategyConfig, Derived>::recordStats(StatsRecorder &sr,
                                                                 std::chrono::duration<float, std::milli> elapsedMS) {
  StrategySubsetting<StrategyConfig, Derived>::recordStats(sr, elapsedMS);
  sr.add("GPU Mode", GPUModeNames[mode]);
  sr.add("GPU Kernels", gpuRootData->kernelsExecuted);
  sr.add("GPU FPS", (float)gpuRootData->kernelsExecuted / (elapsedMS.count() / 1000.0));
  if (gpuRootData->gpuInterface && gpuRootData->gpuInterface->gpuAvailable()) {
    for (const auto &a : gpuRootData->gpuInterface->getGPUInfo()) {
      sr.add(a.first, a.second);
    }
  }
}

template <typename CodewordT>
StrategySubsettingGPURootData<CodewordT>::~StrategySubsettingGPURootData() {
  delete gpuInterface;
  gpuInterface = nullptr;
}

// --------------------------------------------------------------------------------------------------------------------
// Knuth

template <typename StrategyConfig>
size_t StrategyKnuth<StrategyConfig>::computeSubsetScore() {
  int largestSubsetSize = 0;  // Maximum number of codewords that could be retained by using this guess
  for (auto &s : this->ssHolder->subsetSizes) {
    if (s > largestSubsetSize) {
      largestSubsetSize = s;
    }
    s = 0;
  }
  // Invert largestSubsetSize, and return the minimum number of codewords that could be eliminated by using this guess
  return this->possibleSolutions.size() - largestSubsetSize;
}

// --------------------------------------------------------------------------------------------------------------------
// Most Parts

template <typename StrategyConfig>
size_t StrategyMostParts<StrategyConfig>::computeSubsetScore() {
  int totalUsedSubsets = 0;
  for (auto &s : this->ssHolder->subsetSizes) {
    if (s > 0) {
      totalUsedSubsets++;
    }
    s = 0;
  }
  return totalUsedSubsets;
}

// --------------------------------------------------------------------------------------------------------------------
// Expected Size

template <typename StrategyConfig>
size_t StrategyExpectedSize<StrategyConfig>::computeSubsetScore() {
  double expectedSize = 0;
  for (auto &s : this->ssHolder->subsetSizes) {
    if (s > 0) {
      expectedSize += ((double)s * (double)s) / (double)this->possibleSolutions.size();
    }
    s = 0;
  }
  return -round(expectedSize * 10'000'000'000'000'000.0);  // 16 digits of precision
}

// --------------------------------------------------------------------------------------------------------------------
// Entropy

template <typename StrategyConfig>
size_t StrategyEntropy<StrategyConfig>::computeSubsetScore() {
  double entropySum = 0.0;
  for (auto &s : this->ssHolder->subsetSizes) {
    if (s > 0) {
      double pi = (double)s / this->possibleSolutions.size();
      entropySum -= pi * log(pi);
    }
    s = 0;
  }
  return round(entropySum * 10'000'000'000'000'000.0);  // 16 digits of precision
}
