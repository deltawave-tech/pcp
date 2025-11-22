// In src/network/capnp_bridge.h
#ifndef CAPNP_BRIDGE_H
#define CAPNP_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct declarations for type safety
typedef struct CapnpMessageBuilder CapnpMessageBuilder;
typedef struct CapnpMessageReader CapnpMessageReader;

// --- WorkerPayload ---
CapnpMessageBuilder* new_worker_payload_builder();
void set_worker_payload_params(CapnpMessageBuilder* builder, const uint8_t* data, size_t size);
void set_worker_payload_input_ids(CapnpMessageBuilder* builder, const uint8_t* data, size_t size);
void set_worker_payload_targets(CapnpMessageBuilder* builder, const uint8_t* data, size_t size);

// --- ShepherdPayload ---
CapnpMessageBuilder* new_shepherd_payload_builder();
void set_shepherd_payload_params(CapnpMessageBuilder* builder, const uint8_t* data, size_t size);
void set_shepherd_payload_loss(CapnpMessageBuilder* builder, float loss);

// --- Common Serialization Functions ---
size_t get_message_size(CapnpMessageBuilder* builder);
size_t message_to_bytes(CapnpMessageBuilder* builder, uint8_t* buffer, size_t buffer_size);
void free_builder(CapnpMessageBuilder* builder);

// --- Deserialization ---
CapnpMessageReader* new_message_reader(const uint8_t* data, size_t size);
int get_worker_payload_params(CapnpMessageReader* reader, const uint8_t** data, size_t* size);
int get_worker_payload_input_ids(CapnpMessageReader* reader, const uint8_t** data, size_t* size);
int get_worker_payload_targets(CapnpMessageReader* reader, const uint8_t** data, size_t* size);
int get_shepherd_payload_params(CapnpMessageReader* reader, const uint8_t** data, size_t* size);
int get_shepherd_payload_loss(CapnpMessageReader* reader, float* loss);
void free_reader(CapnpMessageReader* reader);

#ifdef __cplusplus
}
#endif

#endif // CAPNP_BRIDGE_H