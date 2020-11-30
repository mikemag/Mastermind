// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

std::string commaString(float f) {
  std::locale comma_locale(std::locale(), new comma_numpunct());
  std::stringstream ss;
  ss.imbue(comma_locale);
  ss << std::setprecision(4) << std::fixed << f;
  return ss.str();
}
