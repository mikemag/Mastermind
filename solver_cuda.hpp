// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "solver.hpp"

// CUDA implementation for playing all games at once
//
// TODO: this needs a lot of notes and docs consolidated

template <typename SolverConfig_>
class SolverCUDA : public Solver {
  using CodewordT = typename SolverConfig_::CodewordT;

 public:
  using SolverConfig = SolverConfig_;

  void playAllGames(uint32_t packedInitialGuess) override;
};

#include "solver_cuda.inl"
