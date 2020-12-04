// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

enum Algo {
  FirstOne,      // Pick the first of the remaining choices.
  Random,        // Pick any of the remaining choices.
  Knuth,         // Pick the one that will eliminate the most remaining choices.
  MostParts,     // Maximize the number of scores at each round.
  ExpectedSize,  // Minimize the expected size of the remaining choices.
  Entropy,       // Pick the maximum entropy guess.
};

enum ComputeKernelBufferIndices {
  BufferIndexAllCodewords = 0,
  BufferIndexAllCodewordsColors = 1,
  BufferIndexPossibleSolutionsCount = 2,
  BufferIndexPossibleSolutions = 3,
  BufferIndexPossibleSolutionsColors = 4,
  BufferIndexScores = 5,
  BufferIndexRemainingIsPossibleSolution = 6,
  BufferIndexFullyDiscriminatingCodewords = 7,
};
