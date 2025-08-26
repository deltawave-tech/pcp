#ifndef CAPNP_BRIDGE_H
#define CAPNP_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct to hide C++ implementation details from Zig
typedef struct CapnpMessageBuilder CapnpMessageBuilder;
typedef struct CapnpMessageReader CapnpMessageReader;

// --- WorkerPayload Serialization (Zig -> C++) ---
CapnpMessageBuilder* new_worker_payload_builder();
void set_worker_payload_params(CapnpMessageBuilder* builder, const uint8_t* data, size_t size);

// --- ShepherdPayload Serialization (Zig -> C++) ---
CapnpMessageBuilder* new_shepherd_payload_builder();
void set_shepherd_payload_params(CapnpMessageBuilder* builder, const uint8_t* data, size_t size);
void set_shepherd_payload_loss(CapnpMessageBuilder* builder, float loss);

// --- Common Serialization Function ---
// Serializes the builder's content and writes it into the provided buffer.
// Returns the number of bytes written.
size_t message_to_bytes(CapnpMessageBuilder* builder, uint8_t* buffer, size_t buffer_size);
size_t get_message_size(CapnpMessageBuilder* builder);
void free_builder(CapnpMessageBuilder* builder);


// --- Deserialization (C++ -> Zig) ---
CapnpMessageReader* new_message_reader(const uint8_t* data, size_t size);

// Returns 1 on success, 0 on failure.
int get_worker_payload_params(CapnpMessageReader* reader, const uint8_t** data, size_t* size);
int get_shepherd_payload_params(CapnpMessageReader* reader, const uint8_t** data, size_t* size);
int get_shepherd_payload_loss(CapnpMessageReader* reader, float* loss);

void free_reader(CapnpMessageReader* reader);

#ifdef __cplusplus
}
#endif

#endif // CAPNP_BRIDGE_H