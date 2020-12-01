// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

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
