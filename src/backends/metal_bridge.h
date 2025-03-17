#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Modified custom version for easier FFI
bool hasMetalDevice(void);

// Metal device and context
id<MTLDevice> MTLCreateSystemDefaultDevice(void);
id<MTLCommandQueue> MTLDevice_newCommandQueue(id<MTLDevice> device);
id<MTLLibrary> MTLDevice_newLibraryWithSource(id<MTLDevice> device, NSString *source, NSDictionary *options, NSError **err);
id<MTLFunction> MTLLibrary_newFunctionWithName(id<MTLLibrary> library, NSString *name);
id<MTLComputePipelineState> MTLDevice_newComputePipelineStateWithFunction(id<MTLDevice> device, id<MTLFunction> function, NSError **err);
id<MTLBuffer> MTLDevice_newBufferWithBytes(id<MTLDevice> device, const void *bytes, NSUInteger length, MTLResourceOptions options);
id<MTLBuffer> MTLDevice_newBufferWithLength(id<MTLDevice> device, NSUInteger length, MTLResourceOptions options);
void* MTLBuffer_contents(id<MTLBuffer> buffer);
id<MTLCommandBuffer> MTLCommandQueue_commandBuffer(id<MTLCommandQueue> queue);
id<MTLComputeCommandEncoder> MTLCommandBuffer_computeCommandEncoder(id<MTLCommandBuffer> buffer);
void MTLComputeCommandEncoder_setComputePipelineState(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> state);
void MTLComputeCommandEncoder_setBuffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index);
NSUInteger MTLComputePipelineState_maxTotalThreadsPerThreadgroup(id<MTLComputePipelineState> state);
NSUInteger MTLComputePipelineState_threadExecutionWidth(id<MTLComputePipelineState> state);
void MTLComputeCommandEncoder_dispatchThreads(id<MTLComputeCommandEncoder> encoder, MTLSize threads, MTLSize threadsPerThreadgroup);
void MTLComputeCommandEncoder_endEncoding(id<MTLComputeCommandEncoder> encoder);
void MTLCommandBuffer_commit(id<MTLCommandBuffer> buffer);
void MTLCommandBuffer_waitUntilCompleted(id<MTLCommandBuffer> buffer);

// NSString utilities
NSString* NSString_alloc(void);
NSString* NSString_initWithUTF8String(NSString *self, const char *cString);
void NSString_release(NSString *self);

#endif // METAL_BRIDGE_H