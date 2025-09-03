# src/network/protocol.capnp
@0xABCD1234EF567890; # Unique file ID

# Message from Shepherd to Worker for starting the inner loop
struct WorkerPayload {
    params @0 :Data;
    inputIds @1 :Data;    # Serialized input token IDs
    targets @2 :Data;     # Serialized target token IDs
}

# Message from Worker back to Shepherd after inner loop completion
struct ShepherdPayload {
    updatedParams @0 :Data;
    loss @1 :Float32;
}