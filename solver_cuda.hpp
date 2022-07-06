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

template <typename SolverConfig>
class SolverCUDA : public Solver<SolverConfig> {
  using CodewordT = typename SolverConfig::CodewordT;

 public:
  void playAllGames() override;



};

#include "solver_cuda.inl"
