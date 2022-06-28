// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "compute_kernel_constants.h"
#import "metal_gpu_interface.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// TODO: the Metal impl is completely broken now. I didn't want to keep hacking up the current work advancing w/ CUDA
// just to keep this working. I need to get back to this at some point and fix it all up, implement the new things in
// the Metal kernels, etc.

@implementation MetalGPUInterface {
  uint _mPinCount;

  id<MTLDevice> _mDevice;
  id<MTLComputePipelineState> _mComputeKernelFunctionPSO;
  id<MTLCommandQueue> _mCommandQueue;

  int _mAllCodewordsCount;
  id<MTLBuffer> _mBufferAllCodewords;
  id<MTLBuffer> _mBufferAllCodewordsColors;

  id<MTLBuffer> _mBufferPossibleSolutionsCount;
  id<MTLBuffer> _mBufferPossibleSolutions;
  id<MTLBuffer> _mBufferPossibleSolutionsColors;

  id<MTLBuffer> _mBufferScores;
  id<MTLBuffer> _mBufferRemainingIsPossibleSolution;

  id<MTLBuffer> _mBufferFullyDiscriminatingCodewords;
}

- (instancetype)initWithPinCount:(uint)pinCount totalCodewords:(uint)totalCodewords kernelName:(NSString *)kernelName {
  self = [super init];
  if (self) {
    NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();

    printf("GPU devices available:\n\n");
    _mDevice = nil;
    for (id<MTLDevice> device in availableDevices) {
      printf("GPU name: %s\n", device.name.UTF8String);
      printf("Max threads per threadgroup: %lu\n", (unsigned long)device.maxThreadsPerThreadgroup.width);
      printf("Max threadgroup memory length: %lu\n", (unsigned long)device.maxThreadgroupMemoryLength);
      printf("Max buffer length: %lu\n", (unsigned long)device.maxBufferLength);
      printf("\n");

      if ((_mDevice == nil) || (!device.isLowPower && [device supportsFamily:MTLGPUFamilyMac2])) {
        _mDevice = device;

        _gpuInfo = @{
          @"GPU Name" : device.name,
          @"GPU Shared Memory per block" :
              [NSString stringWithFormat:@"%lu", (unsigned long)device.maxThreadgroupMemoryLength],
          @"GPU Threads per Block" :
              [NSString stringWithFormat:@"%lu", (unsigned long)device.maxThreadsPerThreadgroup.width]
        };
      }
    }

    if (_mDevice == nil) {
      printf("No suitable GPU found!\n");
      return nil;
    }

    printf("Using GPU: %s\n\n", _mDevice.name.UTF8String);

    if (![_mDevice supportsFamily:MTLGPUFamilyMac2]) {
      printf("WARNING: selected GPU does not support MTLGPUFamilyMac2, some opts will be disabled.\n\n");
      kernelName = [kernelName stringByAppendingString:@"_no2"];
    }

    id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
    if (defaultLibrary == nil) {
      NSLog(@"Failed to find the default library.");
      exit(-1);
    }

    // Lookup the specialized kernel using the pin count in the name
    _mPinCount = pinCount;
    id<MTLFunction> computeKernelFunction =
        [defaultLibrary newFunctionWithName:[NSMutableString stringWithFormat:@"%@_%d", kernelName, _mPinCount]];
    if (computeKernelFunction == nil) {
      NSLog(@"Failed to find kernel function: %@_%d", kernelName, _mPinCount);
      exit(-1);
    }

    NSError *error = nil;
    _mComputeKernelFunctionPSO = [_mDevice newComputePipelineStateWithFunction:computeKernelFunction error:&error];
    if (_mComputeKernelFunctionPSO == nil) {
      NSLog(@"Failed to created pipeline state object, error %@.", error);
      exit(-1);
    }

    _mCommandQueue = [_mDevice newCommandQueue];
    if (_mCommandQueue == nil) {
      NSLog(@"Failed to find the command queue.");
      exit(-1);
    }

    [self createBuffers:totalCodewords executionWidth:_mComputeKernelFunctionPSO.threadExecutionWidth];
  }
  return self;
}

// TODO: consider merging the two small buffers into a single buffer w/ a struct.
- (void)createBuffers:(uint)maxCodewords executionWidth:(NSUInteger)executionWidth {
  _mBufferAllCodewords = [_mDevice newBufferWithLength:maxCodewords * sizeof(uint32_t)
                                               options:MTLResourceStorageModeManaged];
  _mBufferAllCodewordsColors = [_mDevice newBufferWithLength:maxCodewords * sizeof(unsigned __int128)
                                                     options:MTLResourceStorageModeManaged];

  _mBufferPossibleSolutionsCount = [_mDevice newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  _mBufferPossibleSolutions =
      [_mDevice newBufferWithLength:maxCodewords * sizeof(uint32_t)
                            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
  _mBufferPossibleSolutionsColors =
      [_mDevice newBufferWithLength:maxCodewords * sizeof(unsigned __int128)
                            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];

  _mBufferScores = [_mDevice newBufferWithLength:maxCodewords * sizeof(uint32_t) options:MTLResourceStorageModeShared];
  _mBufferRemainingIsPossibleSolution = [_mDevice newBufferWithLength:maxCodewords * sizeof(bool)
                                                              options:MTLResourceStorageModeShared];

  if ([_mDevice supportsFamily:MTLGPUFamilyMac2]) {
    _mBufferFullyDiscriminatingCodewords =
        [_mDevice newBufferWithLength:((maxCodewords / executionWidth) + 1) * sizeof(uint32_t)
                              options:MTLResourceStorageModeShared];
  } else {
    _mBufferFullyDiscriminatingCodewords = nil;
  }
}

- (uint32_t *)getAllCodewordsBuffer {
  return _mBufferAllCodewords.contents;
}

- (unsigned __int128 *)getAllCodewordsColorsBuffer {
  return _mBufferAllCodewordsColors.contents;
}

- (void)setAllCodewordsCount:(uint32_t)count {
  _mAllCodewordsCount = count;
}

- (void)syncAllCodewords:(uint32_t)count {
  [_mBufferAllCodewords didModifyRange:NSMakeRange(0, count * sizeof(uint32_t))];
  [_mBufferAllCodewordsColors didModifyRange:NSMakeRange(0, count * sizeof(unsigned __int128))];
}

- (uint32_t *)getPossibleSolutionssBuffer {
  return _mBufferPossibleSolutions.contents;
}

- (unsigned __int128 *)getPossibleSolutionsColorsBuffer {
  return _mBufferPossibleSolutionsColors.contents;
}

- (void)setPossibleSolutionsCount:(uint32_t)count {
  *((uint32_t *)_mBufferPossibleSolutionsCount.contents) = count;
}

- (void)sendComputeCommand {
  bool capture = false; // Debugging
  if (capture) {
    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = _mDevice;
    NSError *error;
    if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
      NSLog(@"Failed to start capture, error %@", error);
    }
  }

  id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
  assert(commandBuffer != nil);
  id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
  assert(computeEncoder != nil);
  [self encodeComputeCommand:computeEncoder];
  [computeEncoder endEncoding];

  // Execute the command
  [commandBuffer commit];

  [commandBuffer waitUntilCompleted];

  MTLCommandBufferStatus s = [commandBuffer status];
  if (s == MTLCommandBufferStatusError) {
    NSLog(@"Compute kernel failed! PS Size=%d Error: %@", *((uint32_t *)_mBufferPossibleSolutionsCount.contents),
          [commandBuffer error]);
    exit(-1);
  }

  if (capture) {
    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    [captureManager stopCapture];
  }
}

- (void)encodeComputeCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
  [computeEncoder setComputePipelineState:_mComputeKernelFunctionPSO];
  [computeEncoder setBuffer:_mBufferAllCodewords offset:0 atIndex:BufferIndexAllCodewords];
  [computeEncoder setBuffer:_mBufferAllCodewordsColors offset:0 atIndex:BufferIndexAllCodewordsColors];
  [computeEncoder setBuffer:_mBufferPossibleSolutionsCount offset:0 atIndex:BufferIndexPossibleSolutionsCount];
  [computeEncoder setBuffer:_mBufferPossibleSolutions offset:0 atIndex:BufferIndexPossibleSolutions];
  [computeEncoder setBuffer:_mBufferPossibleSolutionsColors offset:0 atIndex:BufferIndexPossibleSolutionsColors];
  [computeEncoder setBuffer:_mBufferScores offset:0 atIndex:BufferIndexScores];
  [computeEncoder setBuffer:_mBufferRemainingIsPossibleSolution
                     offset:0
                    atIndex:BufferIndexRemainingIsPossibleSolution];

  if (_mBufferFullyDiscriminatingCodewords) {
    memset(_mBufferFullyDiscriminatingCodewords.contents, 0, _mBufferFullyDiscriminatingCodewords.length);
    [computeEncoder setBuffer:_mBufferFullyDiscriminatingCodewords
                       offset:0
                      atIndex:BufferIndexFullyDiscriminatingCodewords];
  }

  MTLSize gridSize = MTLSizeMake(_mAllCodewordsCount, 1, 1);

  // 64 covers two SIMD groups and thus one CU on my current GPU. They all ought to be able share the threadgroup
  // memory well.
  NSUInteger targetThreadGroupSize = 64;
  if (targetThreadGroupSize > _mAllCodewordsCount) {
    targetThreadGroupSize = _mAllCodewordsCount;
  }
  MTLSize threadgroupSize = MTLSizeMake(targetThreadGroupSize, 1, 1);

  // NB: matches the def in the compute kernel.
  const int totalScores = ((_mPinCount * (_mPinCount + 3)) / 2) + 1;
  [computeEncoder setThreadgroupMemoryLength:targetThreadGroupSize * sizeof(uint32_t) * totalScores atIndex:0];

  // Encode the compute command.
  [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

// Access to output buffers
- (uint32_t *)getScores {
  return (uint32_t *)_mBufferScores.contents;
}
- (bool *)getRemainingIsPossibleSolution {
  return (bool *)_mBufferRemainingIsPossibleSolution.contents;
}

- (uint32_t *)getFullyDiscriminatingCodewords:(uint32_t *)count {
  if (_mBufferFullyDiscriminatingCodewords) {
    *count = (uint32_t)(_mBufferFullyDiscriminatingCodewords.length / sizeof(uint32_t));
    return (uint32_t *)_mBufferFullyDiscriminatingCodewords.contents;
  } else {
    *count = 0;
    return NULL;
  }
}

- (NSDictionary *)getGPUInfo {
  return _gpuInfo;
}

@end
