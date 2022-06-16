// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MetalGPUInterface : NSObject
- (instancetype)initWithPinCount:(uint)pinCount totalCodewords:(uint)totalCodewords kernelName:(NSString *)kernelName;

- (uint32_t *)getAllCodewordsBuffer;
- (unsigned __int128 *)getAllCodewordsColorsBuffer;
- (void)setAllCodewordsCount:(uint32_t)count;
- (void)syncAllCodewords:(uint32_t)count;

- (uint32_t *)getPossibleSolutionssBuffer;
- (unsigned __int128 *)getPossibleSolutionsColorsBuffer;
- (void)setPossibleSolutionsCount:(uint32_t)count;

- (void)sendComputeCommand;

- (uint32_t *)getScores;
- (bool *)getRemainingIsPossibleSolution;

- (uint32_t *)getFullyDiscriminatingCodewords:(uint32_t *)count;

- (NSString *)getGPUName;
@end

NS_ASSUME_NONNULL_END