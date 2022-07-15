// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string_view>
#include <tuple>

// Sets up counters with names, descriptions, and most importantly a zero-based index. Used to have arrays of counters
// on CPU and GPU which match, can be used on either size, and added together at the end.
struct CounterDescriptor {
  int index;
  const char* name;
  const char* desc;
};

template <std::size_t size>
struct CounterDescriptors {
  std::array<CounterDescriptor, size> descs;

  constexpr CounterDescriptors(std::initializer_list<std::tuple<const char*, const char*>> l) : descs() {
    for (int i = 0; i < l.size(); i++) {
      descs[i] = CounterDescriptor{i, get<0>(data(l)[i]), get<1>(data(l)[i])};
    }
  }
};

// nb: stuck on C++17 right now w/ CUDA, so copy in the constexpr std::find_if.
template <class InputIt, class UnaryPredicate>
constexpr InputIt constexpr_find_if(InputIt first, InputIt last, UnaryPredicate p) {
  for (; first != last; ++first) {
    if (p(*first)) {
      return first;
    }
  }
  return last;
}

template <std::size_t S>
constexpr static auto find_counter(const CounterDescriptors<S>& a, const char* counterName) {
  return constexpr_find_if(
             a.descs.begin(), a.descs.end(),
             [counterName](const CounterDescriptor& c) { return std::string_view(c.name) == counterName; })
      ->index;
};

namespace counters_tests {
constexpr static CounterDescriptors<2> testCA{{{"hi", "hi desc"}, {"hi2", "hi2 desc"}}};
static_assert(testCA.descs[1].index == 1);
static_assert(find_counter(testCA, "hi") == 0);
static_assert(find_counter(testCA, "hi2") == 1);
}  // namespace counters_tests