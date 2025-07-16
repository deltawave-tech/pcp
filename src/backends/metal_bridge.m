#import "metal_bridge.h"

// Simple function for checking if Metal is available
bool hasMetalDevice(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        bool result = (device != nil);
        NSLog(@"Metal device check: %@", result ? @"Found" : @"Not found");
        return result;
    }
}

// Custom implementation for safer Metal device creation
// This function needs to be marked with __attribute__((visibility("default")))
// to ensure it's exported correctly for Zig FFI
// Wrapper for Metal device creation - renamed to avoid conflict with system function
__attribute__((visibility("default")))
id<MTLDevice> MTL_CreateSystemDefaultDevice(void) {
    @autoreleasepool {
        // Use the system Metal function directly from the framework  
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            NSLog(@"Metal device created successfully: %@", device.name);
        } else {
            NSLog(@"Failed to create Metal device!");
        }
        return device;
    }
}

id<MTLCommandQueue> MTLDevice_newCommandQueue(id<MTLDevice> device) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue) {
            NSLog(@"Command queue created successfully");
        } else {
            NSLog(@"Failed to create command queue");
        }
        return queue;
    }
}

id<MTLLibrary> MTLDevice_newLibraryWithSource(id<MTLDevice> device, NSString *source, NSDictionary *options, NSError **err) {
    @autoreleasepool {
        MTLCompileOptions* compileOptions = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:compileOptions error:err];
        if (library) {
            NSLog(@"Metal library created successfully");
        } else if (err && *err) {
            NSLog(@"Failed to create Metal library: %@", [*err localizedDescription]);
        } else {
            NSLog(@"Failed to create Metal library (unknown error)");
        }
        return library;
    }
}

id<MTLFunction> MTLLibrary_newFunctionWithName(id<MTLLibrary> library, NSString *name) {
    @autoreleasepool {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (function) {
            NSLog(@"Metal function '%@' created successfully", name);
        } else {
            NSLog(@"Failed to create Metal function: %@", name);
        }
        return function;
    }
}

id<MTLComputePipelineState> MTLDevice_newComputePipelineStateWithFunction(id<MTLDevice> device, id<MTLFunction> function, NSError **err) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:err];
        if (pipelineState) {
            NSLog(@"Metal pipeline state created successfully");
        } else if (err && *err) {
            NSLog(@"Failed to create Metal pipeline state: %@", [*err localizedDescription]);
        } else {
            NSLog(@"Failed to create Metal pipeline state (unknown error)");
        }
        return pipelineState;
    }
}

id<MTLBuffer> MTLDevice_newBufferWithBytes(id<MTLDevice> device, const void *bytes, NSUInteger length, MTLResourceOptions options) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithBytes:bytes length:length options:options];
        if (buffer) {
            NSLog(@"Metal buffer created successfully with %lu bytes", (unsigned long)length);
        } else {
            NSLog(@"Failed to create Metal buffer with %lu bytes", (unsigned long)length);
        }
        return buffer;
    }
}

id<MTLBuffer> MTLDevice_newBufferWithLength(id<MTLDevice> device, NSUInteger length, MTLResourceOptions options) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:options];
        if (buffer) {
            NSLog(@"Metal buffer created successfully with %lu bytes", (unsigned long)length);
        } else {
            NSLog(@"Failed to create Metal buffer with %lu bytes", (unsigned long)length);
        }
        return buffer;
    }
}

void* MTLBuffer_contents(id<MTLBuffer> buffer) {
    return [buffer contents];
}

NSUInteger MTLBuffer_length(id<MTLBuffer> buffer) {
    return [buffer length];
}

id<MTLCommandBuffer> MTLCommandQueue_commandBuffer(id<MTLCommandQueue> queue) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        if (cmdBuffer) {
            NSLog(@"Command buffer created successfully");
        } else {
            NSLog(@"Failed to create command buffer");
        }
        return cmdBuffer;
    }
}

id<MTLComputeCommandEncoder> MTLCommandBuffer_computeCommandEncoder(id<MTLCommandBuffer> buffer) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        if (encoder) {
            NSLog(@"Compute command encoder created successfully");
        } else {
            NSLog(@"Failed to create compute command encoder");
        }
        return encoder;
    }
}

void MTLComputeCommandEncoder_setComputePipelineState(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> state) {
    [encoder setComputePipelineState:state];
}

void MTLComputeCommandEncoder_setBuffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
    [encoder setBuffer:buffer offset:offset atIndex:index];
}

NSUInteger MTLComputePipelineState_maxTotalThreadsPerThreadgroup(id<MTLComputePipelineState> state) {
    return [state maxTotalThreadsPerThreadgroup];
}

NSUInteger MTLComputePipelineState_threadExecutionWidth(id<MTLComputePipelineState> state) {
    return [state threadExecutionWidth];
}

void MTLComputeCommandEncoder_dispatchThreads(id<MTLComputeCommandEncoder> encoder, MTLSize threads, MTLSize threadsPerThreadgroup) {
    @autoreleasepool {
        NSLog(@"Dispatching threads: [%lu, %lu, %lu] with threadgroup size: [%lu, %lu, %lu]",
            (unsigned long)threads.width, (unsigned long)threads.height, (unsigned long)threads.depth,
            (unsigned long)threadsPerThreadgroup.width, (unsigned long)threadsPerThreadgroup.height, (unsigned long)threadsPerThreadgroup.depth);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threadsPerThreadgroup];
    }
}

void MTLComputeCommandEncoder_endEncoding(id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding];
}

void MTLCommandBuffer_commit(id<MTLCommandBuffer> buffer) {
    [buffer commit];
}

void MTLCommandBuffer_waitUntilCompleted(id<MTLCommandBuffer> buffer) {
    [buffer waitUntilCompleted];
}

NSString* NSString_alloc(void) {
    @autoreleasepool {
        NSString* str = [NSString alloc];
        if (str) {
            NSLog(@"NSString allocated successfully");
        } else {
            NSLog(@"Failed to allocate NSString");
        }
        return str;
    }
}

NSString* NSString_initWithUTF8String(NSString *self, const char *cString) {
    @autoreleasepool {
        NSString* str = [self initWithUTF8String:cString];
        if (str) {
            NSLog(@"NSString initialized successfully with: %s", cString);
        } else {
            NSLog(@"Failed to initialize NSString with: %s", cString);
        }
        return str;
    }
}

// With ARC, we don't need explicit release
void NSString_release(NSString *self) {
    // Under ARC, this is a no-op
    // The string will be released automatically when needed
    NSLog(@"NSString release called (no-op under ARC)");
}