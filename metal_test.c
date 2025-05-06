// Simple C program to test Metal device creation
#include <stdio.h>
#include <Metal/Metal.h>

int main() {
    printf("=== C Metal Device Test ===\n\n");
    
    // Try to create a Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    if (device) {
        printf("SUCCESS: Metal device created: %s\n", [[device name] UTF8String]);
    } else {
        printf("ERROR: Failed to create Metal device\n");
    }
    
    return 0;
}
