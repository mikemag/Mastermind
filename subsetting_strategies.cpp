// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

using namespace std;

// --------------------------------------------------------------------------------------------------------------------
// Base for all subsetting strategies

template <uint8_t p, uint8_t c, bool log>
Codeword<p, c> StrategySubsetting<p, c, log>::selectNextGuess() {
  // Add the last guess from the list of used codewords.
  usedCodewords = savedUsedCodewords;
  usedCodewords.emplace_back(this->guess.packedCodeword());

  Codeword<p, c> bestGuess;
  size_t bestScore = 0;
  bool bestIsPossibleSolution = false;
  int allCount = 0;  // For metrics only
  for (const auto &g : Codeword<p, c>::getAllCodewords()) {
    bool isPossibleSolution = false;
    for (const auto &ps : this->possibleSolutions) {
      Score r = g.score(ps);
      subsetSizes[r.result]++;
      if (r == Codeword<p, c>::winningScore) {
        isPossibleSolution = true;  // Remember if this guess is in the set of possible solutions
      }
    }
    this->scoreCounterCPU += this->possibleSolutions.size();
    if (Strategy<p, c, log>::enableSmallPSMetrics) allCount++;

    // Shortcut small sets. Note, we've already done this check for possible solutions.
    if (Strategy<p, c, log>::enableSmallPSShortcut && !isPossibleSolution &&
        this->possibleSolutions.size() <= Strategy<p, c, log>::totalScores) {
      int subsetsCount = computeTotalSubsets();
      if (subsetsCount == this->possibleSolutions.size()) {
        if (log) {
          cout << "Selecting fully discriminating guess: " << g << ", subsets: " << subsetsCount << endl;
        }
        fill(begin(subsetSizes), end(subsetSizes), 0);  // Done implicitly in computeSubsetScore, which we're skipping
        ++Strategy<p, c, log>::smallPSInnerShortcuts;
        Strategy<p, c, log>::smallPSInnerScoresSkipped +=
            (Codeword<p, c>::getAllCodewords().size() - allCount) * this->possibleSolutions.size();
        return g;
      }
      ++Strategy<p, c, log>::smallPSInnerWasted;
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

  if (log) {
    cout << "Selecting best guess: " << bestGuess << "\tscore: " << bestScore << endl;
  }
  return bestGuess;
}

template <uint8_t p, uint8_t c, bool log>
int StrategySubsetting<p, c, log>::computeTotalSubsets() {
  int totalSubsets = 0;
  for (auto s : subsetSizes) {
    if (s > 0) {
      totalSubsets++;
    }
  }
  return totalSubsets;
}

// --------------------------------------------------------------------------------------------------------------------
// Base for all subsetting GPU strategies

template <uint8_t p, uint8_t c, bool log>
Codeword<p, c> StrategySubsettingGPU<p, c, log>::selectNextGuess() {
  // Cut off "small" work and just do it on the CPU.
  if (mode == CPU ||
      (mode == Both && Codeword<p, c>::getAllCodewords().size() * this->possibleSolutions.size() < 4 * 1024)) {
    return StrategySubsetting<p, c, log>::selectNextGuess();
  }

  // Add the last guess from the list of used codewords.
  this->usedCodewords = this->savedUsedCodewords;
  this->usedCodewords.emplace_back(this->guess.packedCodeword());

  // Pull out the codewords and colors into individual arrays
  // TODO: if this were separated into these two arrays throughout the Strategy then we could use a target-optimized
  //  memcpy to blit them into the buffers, which would be much faster.
  uint32_t *psw = gpuInterface->getPossibleSolutionssBuffer();
  unsigned __int128 *psc = gpuInterface->getPossibleSolutionsColorsBuffer();
  for (int i = 0; i < this->possibleSolutions.size(); i++) {
    psw[i] = this->possibleSolutions[i].packedCodeword();
    psc[i] = this->possibleSolutions[i].packedColors8();
  }
  gpuInterface->setPossibleSolutionsCount((uint32_t)this->possibleSolutions.size());

  gpuInterface->sendComputeCommand();
  kernelsExecuted++;

  uint32_t *maxScoreCounts = gpuInterface->getScores();
  bool *remainingIsPossibleSolution = gpuInterface->getRemainingIsPossibleSolution();

  Codeword<p, c> bestGuess;
  size_t bestScore = 0;
  bool bestIsPossibleSolution = false;

  for (int i = 0; i < Codeword<p, c>::getAllCodewords().size(); i++) {
    size_t score = maxScoreCounts[i];
    if (score > bestScore || (!bestIsPossibleSolution && remainingIsPossibleSolution[i] && score == bestScore)) {
      auto &codeword = Codeword<p, c>::getAllCodewords()[i];
      if (find(this->usedCodewords.cbegin(), this->usedCodewords.cend(), codeword.packedCodeword()) !=
          this->usedCodewords.end()) {
        continue;  // Ignore codewords we've already used
      }
      bestScore = score;
      bestGuess = codeword;
      bestIsPossibleSolution = remainingIsPossibleSolution[i];
    }
  }

  this->scoreCounterGPU += Codeword<p, c>::getAllCodewords().size() * this->possibleSolutions.size();

  if (log) {
    cout << "Selecting best guess: " << bestGuess << "\tscore: " << bestScore << " (GPU)" << endl;
  }

  return bestGuess;
}

// This moves all the codewords into the correct device-private buffers just once, since it's a) read only and b)
// quite large.
template <uint8_t p, uint8_t c, bool l>
void StrategySubsettingGPU<p, c, l>::copyAllCodewordsToGPU() {
  if (allCodewordsOnGPU || !gpuInterface->gpuAvailable()) {
    return;
  }

  // Pull out the codewords and color into individual arrays
  uint32_t *acw = gpuInterface->getAllCodewordsBuffer();
  unsigned __int128 *acc = gpuInterface->getAllCodewordsColorsBuffer();
  for (int i = 0; i < Codeword<p, c>::getAllCodewords().size(); i++) {
    acw[i] = Codeword<p, c>::getAllCodewords()[i].packedCodeword();
    acc[i] = Codeword<p, c>::getAllCodewords()[i].packedColors8();
  }
  gpuInterface->setAllCodewordsCount((uint32_t)Codeword<p, c>::getAllCodewords().size());

  // This shoves both buffers over into GPU memory just once, where they remain constant after that. No need to touch
  // them again.
  gpuInterface->syncAllCodewords((uint32_t)Codeword<p, c>::getAllCodewords().size());
  allCodewordsOnGPU = true;
}

template <uint8_t p, uint8_t c, bool l>
void StrategySubsettingGPU<p, c, l>::printStats(chrono::duration<float, milli> elapsedMS) {
  if (mode != CPU && gpuInterface->gpuAvailable()) {
    cout << "GPU kernels executed: " << commaString(kernelsExecuted)
         << "  FPS: " << commaString((float)kernelsExecuted / (elapsedMS.count() / 1000.0)) << endl;
  }
}

// --------------------------------------------------------------------------------------------------------------------
// Knuth

template <uint8_t p, uint8_t c, bool l>
size_t StrategyKnuth<p, c, l>::computeSubsetScore() {
  int largestSubsetSize = 0;  // Maximum number of codewords that could be retained by using this guess
  for (auto &s : this->subsetSizes) {
    if (s > largestSubsetSize) {
      largestSubsetSize = s;
    }
    s = 0;
  }
  // Invert largestSubsetSize, and return the minimum number of codewords that could be eliminated by using this guess
  return this->possibleSolutions.size() - largestSubsetSize;
}

template <uint8_t p, uint8_t c, bool l>
shared_ptr<Strategy<p, c, l>> StrategyKnuth<p, c, l>::createNewMove(Score r, Codeword<p, c> nextGuess) {
  auto next = make_shared<StrategyKnuth<p, c, l>>(nextGuess, this->possibleSolutions, this->usedCodewords);
  next->mode = this->mode;
  return next;
}

// --------------------------------------------------------------------------------------------------------------------
// Most Parts

template <uint8_t p, uint8_t c, bool l>
size_t StrategyMostParts<p, c, l>::computeSubsetScore() {
  int totalUsedSubsets = 0;
  for (auto &s : this->subsetSizes) {
    if (s > 0) {
      totalUsedSubsets++;
    }
    s = 0;
  }
  return totalUsedSubsets;
}

template <uint8_t p, uint8_t c, bool l>
shared_ptr<Strategy<p, c, l>> StrategyMostParts<p, c, l>::createNewMove(Score r, Codeword<p, c> nextGuess) {
  auto next = make_shared<StrategyMostParts<p, c, l>>(nextGuess, this->possibleSolutions, this->usedCodewords);
  next->mode = this->mode;
  return next;
}

// --------------------------------------------------------------------------------------------------------------------
// Expected Size

template <uint8_t p, uint8_t c, bool l>
size_t StrategyExpectedSize<p, c, l>::computeSubsetScore() {
  double expectedSize = 0;
  for (auto &s : this->subsetSizes) {
    if (s > 0) {
      expectedSize += ((double)s * (double)s) / (double)this->possibleSolutions.size();
    }
    s = 0;
  }
  return -round(expectedSize * 10'000'000'000'000'000.0);  // 16 digits of precision
}

template <uint8_t p, uint8_t c, bool l>
shared_ptr<Strategy<p, c, l>> StrategyExpectedSize<p, c, l>::createNewMove(Score r, Codeword<p, c> nextGuess) {
  auto next = make_shared<StrategyExpectedSize<p, c, l>>(nextGuess, this->possibleSolutions, this->usedCodewords);
  next->mode = this->mode;
  return next;
}

// --------------------------------------------------------------------------------------------------------------------
// Entropy

template <uint8_t p, uint8_t c, bool l>
size_t StrategyEntropy<p, c, l>::computeSubsetScore() {
  double entropySum = 0.0;
  for (auto &s : this->subsetSizes) {
    if (s > 0) {
      double pi = (double)s / this->possibleSolutions.size();
      entropySum -= pi * log(pi);
    }
    s = 0;
  }
  return round(entropySum * 10'000'000'000'000'000.0);  // 16 digits of precision
}

template <uint8_t p, uint8_t c, bool l>
shared_ptr<Strategy<p, c, l>> StrategyEntropy<p, c, l>::createNewMove(Score r, Codeword<p, c> nextGuess) {
  auto next = make_shared<StrategyEntropy<p, c, l>>(nextGuess, this->possibleSolutions, this->usedCodewords);
  next->mode = this->mode;
  return next;
}
