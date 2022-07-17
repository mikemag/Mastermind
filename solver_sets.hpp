// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <tuple>

#include "solver.hpp"

namespace solver_sets {

// Helpers to build sets of solvers given lists of pin counts, color counts, and algorithm types. This builds up
// tuples of tuples until we hit the actual Solver types specialized with the right SolverConfigs.
//
// Also helpers to apply a runner type to every Solver in one of these sets. This is how multiple games are actually
// run.
//
// This is all template metaprogramming to allow us to have specialized types all the way down to GPU kernels without
// having a bunch of switch statements somewhere in the middle. Overall, these types recurse to build nested tuples
// with more specialized types, one per input set (like pin or color counts). The runner types recurse back down this
// until they hit Solvers, then we apply a templated functor type to actually instantiate the Solver and run it.
//
// I only dabble in TMP, I'm no expert by any stretch, and I expect there's better ways to do these. I tried to keep
// these short and relatively readable (is any TMP actually readable?!) at the expense of efficiency.

// Aliases to shorten the inputs
template <std::size_t... Is>
using pin_counts = std::integer_sequence<size_t, Is...>;

template <std::size_t... Is>
using color_counts = std::integer_sequence<size_t, Is...>;

template <typename... Ts>
using algo_list = std::tuple<Ts...>;

// ------------------------------------------------------------------------------------------------------------------
// Build combos of solver configs given pins, colors, and algos. Makes a tree of tuples with SolverConfigs at the
// leaves.

template <typename PIs, typename CIs, typename AT, bool LOG>
struct solver_config_list;

// First, apply pins
template <std::size_t... Ps, typename CIs, typename AT, bool LOG>
struct solver_config_list<std::index_sequence<Ps...>, CIs, AT, LOG> {
  using type = std::tuple<solver_config_list<std::integral_constant<std::size_t, Ps>, CIs, AT, LOG>...>;
};

// Second, apply colors
template <std::size_t P, std::size_t... Cs, typename AT, bool LOG>
struct solver_config_list<std::integral_constant<size_t, P>, std::index_sequence<Cs...>, AT, LOG> {
  using type = std::tuple<
      solver_config_list<std::integral_constant<std::size_t, P>, std::integral_constant<std::size_t, Cs>, AT, LOG>...>;
};

// Finally, algos and create the SolverConfig leaves
template <std::size_t P, std::size_t C, typename... As, bool LOG>
struct solver_config_list<std::integral_constant<std::size_t, P>, std::integral_constant<std::size_t, C>,
                          std::tuple<As...>, LOG> {
  using type = std::tuple<SolverConfig<P, C, LOG, As>...>;
};

// ------------------------------------------------------------------------------------------------------------------
// Build solvers given a set of solver configs, making a similar tree as the configs but w/ solvers are the leaves

template <template <class> class SolverT, typename Ts, class Enable = void>
struct build_solvers;

// First, recursively unpack tuples until we find something that's not a tuple
template <template <class> class SolverT, typename... Ts>
struct build_solvers<SolverT, std::tuple<Ts...>,
                     typename std::enable_if<!conjunction_v<is_base_of<SolverConfigBase, Ts>...>>::type> {
  using type = std::tuple<build_solvers<SolverT, typename Ts::type>...>;
};

// Second, build the Solver types given the SolverConfigs we've found at the leaves
template <template <class> class SolverT, typename... Ts>
struct build_solvers<SolverT, std::tuple<Ts...>,
                     typename std::enable_if<std::conjunction_v<std::is_base_of<SolverConfigBase, Ts>...>>::type> {
  using type = std::tuple<SolverT<Ts>...>;
};

// ------------------------------------------------------------------------------------------------------------------
// Run a set of solvers

// Second, apply the functor to each Solver type at the leaves of the tree
template <typename... Ts, typename PlayAllOp,
          typename std::enable_if<conjunction_v<is_base_of<Solver, Ts>...>, bool>::type = true>
static void run_multiple_solvers(const std::tuple<Ts...>&, PlayAllOp playAllOp) {
  const int dummy[] = {0, (playAllOp.template runSolver<Ts>(), 0)...};
  (void)dummy;
}

// First, recursively unpack tuples until we find something that's not a tuple
template <typename... Ts, typename PlayAllOp,
          typename std::enable_if<!conjunction_v<is_base_of<Solver, Ts>...>, bool>::type = true>
constexpr static void run_multiple_solvers(const std::tuple<Ts...>&, PlayAllOp playAllOp) {
  const int dummy[] = {0, (run_multiple_solvers(typename Ts::type{}, playAllOp), 0)...};
  (void)dummy;
}

}  // namespace solver_sets
