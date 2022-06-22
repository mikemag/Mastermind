// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>

using namespace std;

string commaString(float f) {
  locale comma_locale(locale(), new comma_numpunct());
  stringstream ss;
  ss.imbue(comma_locale);
  ss << setprecision(4) << fixed << f;
  return ss.str();
}

void StatsRecorder::writeStats(const std::string &filename) {
  if (runs.empty()) {
    cout << "No stats to write" << endl;
    return;
  }

  cout << "Writing all game stats to " << filename << endl;

  // Common data that we want to show up first, to make the files easier to read quickly.
  vector<string> common = {
      "Pin Count", "Color Count", "Strategy", "GPU Mode", "Initial Guess", "Average Turns", "Max Turns", "Elapsed (s)",
  };

  // TODO: I haven't checked if this is sufficient.
  auto escape = [](string s) {
    if (s.find(',') != std::string::npos) {
      return "\"" + s + "\"";
    }
    return s;
  };

  // Some runs may include stats that others don't, so aggregate the headers from them all so every row as the same
  // number of columns in the same order.
  set<string> headers;
  for (const auto &run : runs) {
    for (const auto &s : run) {
      if (find(begin(common), end(common), s.first) != common.end()) continue;
      headers.insert(s.first);
    }
  }

  ofstream ss(filename);

  // Write header row first
  auto cit = common.begin();
  ss << escape(*cit);
  for (++cit; cit != common.end(); ++cit) {
    ss << "," << escape(*cit);
  }
  for (const auto &h : headers) {
    ss << "," << escape(h);
  }
  for (const auto &a : all) {
    ss << "," << escape(a.first);
  }
  ss << endl;

  // One row per game
  for (auto &run : runs) {
    cit = common.begin();
    ss << escape(run[*cit]);
    for (++cit; cit != common.end(); ++cit) {
      ss << "," << escape(run[*cit]);
    }
    for (const auto &h : headers) {
      ss << "," << escape(run[h]);
    }
    for (const auto &a : all) {
      ss << "," << escape(a.second);
    }
    ss << endl;
  }

  ss.close();
}

// ------------------------------------------------------------------------------------
// Hardware, OS, and CPU info

#if __APPLE__
#include <sys/sysctl.h>
#endif

#if __linux__
#include <libcpuid.h>
#include <sys/sysinfo.h>
#endif

#include <sys/utsname.h>

template <typename T>
string OSInfo::macOSSysctlByName(const string &name) {
  T value;
  size_t size = sizeof(value);
  if (sysctlbyname(name.c_str(), &value, &size, nullptr, 0) < 0) {
    cerr << "sysctlbyname failed for " << name << std::endl;
    exit(-1);
  }
  return to_string(value);
}

#if __APPLE__
template <>
string OSInfo::macOSSysctlByName<std::string>(const string &name) {
  char buffer[1024];
  size_t size = sizeof(buffer);
  if (sysctlbyname(name.c_str(), &buffer, &size, nullptr, 0) < 0) {
    cerr << "sysctlbyname failed for " << name << std::endl;
    exit(-1);
  }
  return {buffer};
}
#endif

OSInfo::OSInfo() {
#if __APPLE__ || __linux__
  utsname uts;
  uname(&uts);
  info["OS Kernel Name"] = uts.sysname;
  info["Machine Name"] = uts.nodename;
  info["OS Kernel Version"] = uts.release;
  info["OS Version String"] = uts.version;
  info["HW CPU Arch"] = uts.machine;
#endif

#if __APPLE__
  info["HW CPU Brand String"] = macOSSysctlByName<string>("machdep.cpu.brand_string");
  info["HW CPU Cacheline Size"] = macOSSysctlByName<uint64_t>("hw.cachelinesize");
  info["HW CPU L1 iCache Size"] = macOSSysctlByName<uint64_t>("hw.l1icachesize");
  info["HW CPU L1 dCache Size"] = macOSSysctlByName<uint64_t>("hw.l1dcachesize");
  info["HW CPU L2 Cache Size"] = macOSSysctlByName<uint64_t>("hw.l2cachesize");
  info["HW CPU L3 Cache Size"] = macOSSysctlByName<uint64_t>("hw.l3cachesize");
  info["HW CPU Physical Count"] = macOSSysctlByName<int32_t>("hw.physicalcpu");
  info["HW CPU Logical Count"] = macOSSysctlByName<int32_t>("hw.logicalcpu");
  info["HW Memory Size"] = macOSSysctlByName<int64_t>("hw.memsize");
  info["HW Model"] = macOSSysctlByName<string>("hw.model");
  info["OS Version"] = macOSSysctlByName<string>("kern.osversion");
  info["OS Product Version"] = macOSSysctlByName<string>("kern.osproductversion");
#endif

#if __linux__
  struct cpu_raw_data_t raw {};
  struct cpu_id_t data {};
  if (cpuid_get_raw_data(&raw) < 0) {
    printf("**FAILED TO GET CPU INFO**: %s\n", cpuid_error());
    return;
  }
  if (cpu_identify(&raw, &data) < 0) {
    printf("**FAILED TO GET CPU INFO**: %s\n", cpuid_error());
    return;
  }

  info["HW CPU Brand String"] = data.brand_str;
  info["HW CPU Cacheline Size"] = to_string(data.l1_cacheline);
  info["HW CPU L1 iCache Size"] = to_string(data.l1_instruction_cache * 1024);
  info["HW CPU L1 dCache Size"] = to_string(data.l1_data_cache * 1024);
  info["HW CPU L2 Cache Size"] = to_string(data.l2_cache * 1024);
  info["HW CPU L3 Cache Size"] = to_string(data.l3_cache * 1024);
  info["HW CPU Physical Count"] = to_string(data.num_cores);
  info["HW CPU Logical Count"] = to_string(data.num_logical_cpus);

  struct sysinfo si {};
  sysinfo(&si);
  info["HW Memory Size"] = to_string(si.totalram);

  std::string token;
  std::ifstream file("/etc/os-release");
  char tmp[1024];
  while (file.getline(tmp, 1024, '=')) {
    if (string(tmp) == "PRETTY_NAME") {
      file.getline(tmp, 1024);
      string s(tmp);
      s.erase(remove(s.begin(), s.end(), '\"'), s.end());
      info["OS Product Version"] = s;
    }
    // Ignore the rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
#endif
}
