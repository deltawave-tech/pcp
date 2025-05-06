// Simple Objective-C program to test Metal device creation
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main() {
    @autoreleasepool {
        NSLog(@"=== Objective-C Metal Device Test ===");
        
        // Try to create a Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        if (device) {
            NSLog(@"SUCCESS: Metal device created: %@", device.name);
        } else {
            NSLog(@"ERROR: Failed to create Metal device");
        }
    }
    
    return 0;
}
