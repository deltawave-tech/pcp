#include "capnp_bridge.h"
#include "protocol.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <kj/std/iostream.h>
#include <kj/array.h>

// Define the opaque structs using C++ types
struct CapnpMessageBuilder {
    capnp::MallocMessageBuilder builder;
};

struct CapnpMessageReader {
    // We use a kj::Array to own the memory for the reader
    kj::Array<capnp::word> buffer;
    capnp::ReaderOptions options;
    capnp::FlatArrayMessageReader reader;

    CapnpMessageReader(kj::Array<capnp::word> buf, capnp::ReaderOptions opts)
        : buffer(kj::mv(buf)), options(opts), reader(buffer, options) {}
};

// --- WorkerPayload ---
CapnpMessageBuilder* new_worker_payload_builder() {
    auto cpp_builder = new CapnpMessageBuilder();
    cpp_builder->builder.initRoot<WorkerPayload>();
    return cpp_builder;
}

void set_worker_payload_params(CapnpMessageBuilder* builder, const uint8_t* data, size_t size) {
    auto root = builder->builder.getRoot<WorkerPayload>();
    root.setParams(kj::ArrayPtr<const kj::byte>(data, size));
}

void set_worker_payload_input_ids(CapnpMessageBuilder* builder, const uint8_t* data, size_t size) {
    auto root = builder->builder.getRoot<WorkerPayload>();
    root.setInputIds(kj::ArrayPtr<const kj::byte>(data, size));
}

void set_worker_payload_targets(CapnpMessageBuilder* builder, const uint8_t* data, size_t size) {
    auto root = builder->builder.getRoot<WorkerPayload>();
    root.setTargets(kj::ArrayPtr<const kj::byte>(data, size));
}

// --- ShepherdPayload ---
CapnpMessageBuilder* new_shepherd_payload_builder() {
    auto cpp_builder = new CapnpMessageBuilder();
    cpp_builder->builder.initRoot<ShepherdPayload>();
    return cpp_builder;
}

void set_shepherd_payload_params(CapnpMessageBuilder* builder, const uint8_t* data, size_t size) {
    auto root = builder->builder.getRoot<ShepherdPayload>();
    root.setUpdatedParams(kj::ArrayPtr<const kj::byte>(data, size));
}

void set_shepherd_payload_loss(CapnpMessageBuilder* builder, float loss) {
    auto root = builder->builder.getRoot<ShepherdPayload>();
    root.setLoss(loss);
}

// --- Common Functions ---
kj::Array<capnp::word> messageToWords(CapnpMessageBuilder* builder) {
    return capnp::messageToFlatArray(builder->builder);
}

size_t get_message_size(CapnpMessageBuilder* builder) {
    return messageToWords(builder).asBytes().size();
}

size_t message_to_bytes(CapnpMessageBuilder* builder, uint8_t* buffer, size_t buffer_size) {
    auto words = messageToWords(builder);
    auto bytes = words.asBytes();
    if (bytes.size() > buffer_size) {
        return 0; // Error: buffer too small
    }
    memcpy(buffer, bytes.begin(), bytes.size());
    return bytes.size();
}

void free_builder(CapnpMessageBuilder* builder) {
    delete builder;
}

// --- Deserialization ---
CapnpMessageReader* new_message_reader(const uint8_t* data, size_t size) {
    // Copy data into a kj::Array so the reader can own it.
    auto buf = kj::heapArray<capnp::word>((size + 7) / 8);
    memcpy(buf.asBytes().begin(), data, size);

    // Default Cap'n Proto traversal limits are too small for our large parameter
    // payloads (tens of MB). Set traversal limit proportional to message size.
    capnp::ReaderOptions options;
    options.traversalLimitInWords = buf.size() * 2;
    return new CapnpMessageReader(kj::mv(buf), options);
}

int get_worker_payload_params(CapnpMessageReader* reader, const uint8_t** data, size_t* size) {
    auto payload = reader->reader.getRoot<WorkerPayload>();
    auto params = payload.getParams();
    *data = params.begin();
    *size = params.size();
    return 1;
}

int get_worker_payload_input_ids(CapnpMessageReader* reader, const uint8_t** data, size_t* size) {
    auto payload = reader->reader.getRoot<WorkerPayload>();
    auto input_ids = payload.getInputIds();
    *data = input_ids.begin();
    *size = input_ids.size();
    return 1;
}

int get_worker_payload_targets(CapnpMessageReader* reader, const uint8_t** data, size_t* size) {
    auto payload = reader->reader.getRoot<WorkerPayload>();
    auto targets = payload.getTargets();
    *data = targets.begin();
    *size = targets.size();
    return 1;
}

int get_shepherd_payload_params(CapnpMessageReader* reader, const uint8_t** data, size_t* size) {
    auto payload = reader->reader.getRoot<ShepherdPayload>();
    auto params = payload.getUpdatedParams();
    *data = params.begin();
    *size = params.size();
    return 1;
}

int get_shepherd_payload_loss(CapnpMessageReader* reader, float* loss) {
    auto payload = reader->reader.getRoot<ShepherdPayload>();
    *loss = payload.getLoss();
    return 1;
}

void free_reader(CapnpMessageReader* reader) {
    delete reader;
}
