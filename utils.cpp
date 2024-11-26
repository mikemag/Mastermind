// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

#include "mastermind_config.h"

using namespace std;
using json = nlohmann::json;

string commaString(float f) {
  locale comma_locale(locale(), new comma_numpunct());
  stringstream ss;
  ss.imbue(comma_locale);
  ss << setprecision(4) << fixed << f;
  return ss.str();
}

std::string hexString(uint32_t v) {
  std::stringstream ss;
  ss << std::hex << v;
  return ss.str();
}

void StatsRecorder::emitSysInfo() {
  std::map<std::string, json> sysInfo;

  for (const auto &a : osInfo.info) {
    sysInfo[a.first] = a.second;
  }

  for (const auto &a : gpuInfo.info) {
    sysInfo[a.first] = a.second;
  }

  sysInfo["Git Branch"] = MASTERMIND_GIT_BRANCH;
  sysInfo["Git Commit Hash"] = MASTERMIND_GIT_COMMIT_HASH;
  sysInfo["Git Commit Date"] = MASTERMIND_GIT_COMMIT_DATE;
  json sysInfoJSON = {{"system_specs", json(sysInfo)}};
  js = ofstream(filename);
  js << "[" << endl;
  js << sysInfoJSON << endl;
  js << flush;
}

StatsRecorder::~StatsRecorder() {
  newRun();
  js << "]" << endl;
  js.close();
}

void StatsRecorder::newRun() {
  if (!run.empty()) {
    if (!js.is_open()) {
      emitSysInfo();
    }

    json runInfo = {{"run", json(run)}};
    js << "," << runInfo << endl << flush;
    run.clear();
  }
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

#if __APPLE__
template <typename T>
T OSInfo::macOSSysctlByName(const string &name, bool optional) {
  T value = 0;
  size_t size = sizeof(value);
  if (!optional && sysctlbyname(name.c_str(), &value, &size, nullptr, 0) < 0) {
    cerr << "sysctlbyname failed for " << name << std::endl;
    exit(EXIT_FAILURE);
  }
  return value;
}

template <>
string OSInfo::macOSSysctlByName<std::string>(const string &name, bool optional) {
  char buffer[1024];
  buffer[0] = 0;
  size_t size = sizeof(buffer);
  if (!optional && sysctlbyname(name.c_str(), &buffer, &size, nullptr, 0) < 0) {
    cerr << "sysctlbyname failed for " << name << std::endl;
    exit(EXIT_FAILURE);
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

  // Use the best perf level to match other systems.
  info["HW CPU L1 iCache Size"] = macOSSysctlByName<uint64_t>("hw.perflevel0.l1icachesize");
  info["HW CPU L1 dCache Size"] = macOSSysctlByName<uint64_t>("hw.perflevel0.l1dcachesize");
  info["HW CPU L2 Cache Size"] = macOSSysctlByName<uint64_t>("hw.perflevel0.l2cachesize");
  info["HW CPU L3 Cache Size"] = macOSSysctlByName<uint64_t>("hw.perflevel0.l3cachesize", true);
  info["HW CPU Physical Count"] = macOSSysctlByName<int32_t>("hw.perflevel0.physicalcpu");
  info["HW CPU Logical Count"] = macOSSysctlByName<int32_t>("hw.perflevel0.logicalcpu");

  // Now grab all perf levels, just for fun
  auto nperflevels = macOSSysctlByName<int>("hw.nperflevels");
  for (auto i = 0; i < nperflevels; i++) {
    string prefix = "hw.perflevel" + to_string(i);
    info["HW CPU Perf Level " + to_string(i) + " L1 iCache Size"] =
        macOSSysctlByName<uint64_t>(prefix + ".l1icachesize");
    info["HW CPU Perf Level " + to_string(i) + " L1 dCache Size"] =
        macOSSysctlByName<uint64_t>(prefix + ".l1dcachesize");
    info["HW CPU Perf Level " + to_string(i) + " L2 Cache Size"] = macOSSysctlByName<uint64_t>(prefix + ".l2cachesize");
    info["HW CPU Perf Level " + to_string(i) + " L3 Cache Size"] =
        macOSSysctlByName<uint64_t>(prefix + ".l3cachesize", true);
    info["HW CPU Perf Level " + to_string(i) + " Physical Count"] = macOSSysctlByName<int32_t>(prefix + ".physicalcpu");
    info["HW CPU Perf Level " + to_string(i) + " Physical Max"] =
        macOSSysctlByName<int32_t>(prefix + ".physicalcpu_max");
    info["HW CPU Perf Level " + to_string(i) + " Logical Count"] = macOSSysctlByName<int32_t>(prefix + ".logicalcpu");
    info["HW CPU Perf Level " + to_string(i) + " Logical Max"] = macOSSysctlByName<int32_t>(prefix + ".logicalcpu_max");
    info["HW CPU Perf Level " + to_string(i) + " CPUs per L2"] = macOSSysctlByName<int32_t>(prefix + ".cpusperl2");
    info["HW CPU Perf Level " + to_string(i) + " CPUs per L3"] =
        macOSSysctlByName<int32_t>(prefix + ".cpusperl3", true);
    info["HW CPU Perf Level " + to_string(i) + " Name"] = macOSSysctlByName<string>(prefix + ".name");
  }

  info["HW Memory Size"] = macOSSysctlByName<int64_t>("hw.memsize");
  info["HW Page Size"] = macOSSysctlByName<int64_t>("hw.pagesize");
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
  info["HW CPU Cacheline Size"] = data.l1_cacheline;
  info["HW CPU L1 iCache Size"] = data.l1_instruction_cache * 1024;
  info["HW CPU L1 dCache Size"] = data.l1_data_cache * 1024;
  info["HW CPU L2 Cache Size"] = data.l2_cache * 1024;
  info["HW CPU L3 Cache Size"] = data.l3_cache * 1024;
  info["HW CPU Physical Count"] = data.num_cores;
  info["HW CPU Logical Count"] = data.num_logical_cpus;

  struct sysinfo si {};
  sysinfo(&si);
  info["HW Memory Size"] = si.totalram;

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

GPUInfo::GPUInfo() { loadDeviceInfo(); }

// Dumping device info from https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery
inline int GPUInfo::_ConvertSMVer2Cores(int major, int minor) {
#if defined(__CUDACC__)
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                     {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                     {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                     {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor,
         nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
#else
  return 0;
#endif
}

// Dumping device info from https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery
void GPUInfo::dumpDeviceInfo() {
#if defined(__CUDACC__)
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }

  int dev = 0;  // TODO: handle multiple GPU's and select the best one.
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  printf("Device %d: \"%s\"\n", dev, deviceProp.name);

  int driverVersion = 0, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

  char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(msg, sizeof(msg),
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f), (unsigned long long)deviceProp.totalGlobalMem);
#else
  snprintf(msg, sizeof(msg), "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
           static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f), (unsigned long long)deviceProp.totalGlobalMem);
#endif
  printf("%s", msg);

  printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n", deviceProp.multiProcessorCount,
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

  printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f,
         deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
  // This is supported in CUDA 5.0 (runtime API device properties)
  printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
  }
#else
  // This only available in CUDA 4.0-4.2 (but these were only exposed in the
  // CUDA Driver API)
  int memoryClock;
  getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
  printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
  int memBusWidth;
  getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
  printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
  int L2CacheSize;
  getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

  if (L2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
  }
#endif

  printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
         deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
         deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n", deviceProp.maxTexture1DLayered[0],
         deviceProp.maxTexture1DLayered[1]);
  printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n", deviceProp.maxTexture2DLayered[0],
         deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

  printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
  printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
  printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n", deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
  printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
         deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0],
         deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch);
  printf("  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment);
  printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
         (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
  printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
  printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
  printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
  printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
  printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
         deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
  printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
  printf("  Device supports Managed Memory:                %s\n", deviceProp.managedMemory ? "Yes" : "No");
  printf("  Device supports Direct Mgd Access From Host:   %s\n",
         deviceProp.directManagedMemAccessFromHost ? "Yes" : "No");
  printf("  Device supports Compute Preemption:            %s\n", deviceProp.computePreemptionSupported ? "Yes" : "No");
  printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
  printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
         deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
  printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID,
         deviceProp.pciDeviceID);

  const char *sComputeMode[] = {
      "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
      "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
      "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
      "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
      "Unknown",
      NULL};
  printf("  Compute Mode:\n");
  printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

  printf("\n");
#endif
}

void GPUInfo::loadDeviceInfo() {
#if defined(__CUDACC__)
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }

  int dev = 0;  // TODO: handle multiple GPU's and select the best one.
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  info["GPU Name"] = string(deviceProp.name);
  hasGPU_ = true;

  int driverVersion = 0, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  info["GPU CUDA Driver Version"] = to_string(driverVersion / 1000) + "." + to_string((driverVersion % 100) / 10);
  info["GPU CUDA Runtime Version"] = to_string(runtimeVersion / 1000) + "." + to_string((runtimeVersion % 100) / 10);

  info["GPU CUDA Capability"] = to_string(deviceProp.major) + "." + to_string(deviceProp.minor);

  info["GPU Global Memory"] = deviceProp.totalGlobalMem;

  info["GPU Multiprocessors"] = deviceProp.multiProcessorCount;
  info["GPU Cores/MP"] = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
  info["GPU Cores"] = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

  info["GPU Max Clock Rate MHz"] = deviceProp.clockRate * 1e-3f;

#if CUDART_VERSION >= 5000
  info["GPU Memory Clock Rate MHz"] = deviceProp.memoryClockRate * 1e-3f;
  info["GPU Memory Bus Width"] = deviceProp.memoryBusWidth;

  if (deviceProp.l2CacheSize) {
    info["GPU L2 Cache Size"] = deviceProp.l2CacheSize;
  }
#endif

  info["GPU Constant Memory"] = deviceProp.totalConstMem;
  info["GPU Shared Memory per block"] = deviceProp.sharedMemPerBlock;
  info["GPU Shared Memory per MP"] = deviceProp.sharedMemPerMultiprocessor;
  info["GPU Registers per block"] = deviceProp.regsPerBlock;
  info["GPU Warp Size"] = deviceProp.warpSize;
  info["GPU Threads per MP"] = deviceProp.maxThreadsPerMultiProcessor;
  info["GPU Threads per Block"] = deviceProp.maxThreadsPerBlock;

#if __linux__
  std::string token;
  std::ifstream file("/proc/driver/nvidia/version");
  char tmp[1024];
  while (file.getline(tmp, 1024, ':')) {
    if (string(tmp) == "NVRM version") {
      file.getline(tmp, 1024);
      string s(tmp);
      // ltrim
      s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
      info["GPU Driver Version"] = s;
    }
    // Ignore the rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
#endif

#endif
}

std::map<int, int> &getACrCache() {
  static std::map<int, int> ACrCache;
  static bool isACrCacheLoaded = false;

  if (!isACrCacheLoaded) {
    try {
      std::ifstream f("ACReduced.json");
      auto j = json::parse(f);
      for (const auto &[key, value] : j.items()) {
        int ki = std::stoi(key);
        ACrCache[ki] = value.get<int>();
      }
      cout << "Loaded ACrCache, " << commaString(ACrCache.size()) << " entries." << endl;
    } catch (json::parse_error &ex) {
      cout << "Unable to load ACrCache." << endl;
    }
    isACrCacheLoaded = true;
  }

  return ACrCache;
}