// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "solver.hpp"

// CPU Reference Implementation
//
// This is a simple impl to serve as a reference for all the others. It's not optimized for speed. Hopefully it's clear.
// More details w/ the impl.

template <typename SolverConfig_>
class SolverReferenceImpl : public Solver {
  using CodewordT = typename SolverConfig_::CodewordT;

 public:
  using SolverConfig = SolverConfig_;

  void playAllGames(uint32_t packedInitialGuess) override;

 private:
  CodewordT nextGuess(const vector<CodewordT>& possibleSolutions, const vector<CodewordT>& usedCodewords);
};

#include "solver_cpu_reference.inl"
