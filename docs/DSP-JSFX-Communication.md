# DSP-JSFX Communication

## Split

Use `gmem` for shared random-access state.

Use `msg_*` for block-resolved event delivery.

## Timing

`msg_*` is block-synchronous. Messages emitted during one `@block` are enqueued at the end of that block and become visible when receivers materialize their next block inbox.

Order is FIFO per sender and channel. Global order across multiple senders is not guaranteed.

## Section validity

- `msg_send`, `msg_sendto`, `msg_recv`, `msg_send_buf`, `msg_sendto_buf`, `msg_recv_buf`, `msg_avail`, `msg_kind`, `msg_length`, `msg_dropped`, `msg_clear`, and peer discovery calls are `@block`-only.
- `comm_join`, `msg_subscribe`, `msg_unsubscribe`, `msg_advertise`, `instance_set_name`, `instance_get_name`, `instance_uid`, `gmem_attach`, and `gmem_attach_size` are valid in `@init`, `@slider`, and `@block`.
- `instance_id` is valid in `@init`, `@slider`, and `@block`.
- Scalar `gmem[index]` reads and writes are valid in DSP sections.
- Bulk `gmem_get/gmem_put/gmem_fill/gmem_zero/gmem_copy` are `@block`-only.


## String input sliders

DSP-JSFX adds a string slider form for instance-local names such as bus/domain names:

```eel
slider1:#bus_name="main"<string>Bus Name

@init
comm_join(#bus_name);
gmem_attach(#bus_name);

@slider
// Lets a live text edit move this instance to another domain/bus.
comm_join(#bus_name);
gmem_attach(#bus_name);
```

Rules:

- the alias should be a string variable such as `#bus_name`
- the default can be quoted or raw text
- the slider UI is a text box, not an automatable audio parameter
- the alias variable stores an opaque string handle usable anywhere a string handle is accepted
- string slider values are saved in plugin state per instance

## Identity and routing

Every instance gets a unique numeric `instance_id()` and a generated UID string via `instance_uid(#str)`.

`instance_set_name("name")` stores a human-readable name handle. It is advisory. Routing uses numeric IDs.

Broadcast send:

```eel
msg_send("tempo", OP_SYNC, bpm, beatpos, playing, 0);
```

Direct send:

```eel
msg_sendto(target_id, "ctl", OP_SET, param, value, 0, 0);
```

## Discovery

Discovery is advisory. Use it for topology inspection and handshakes, not hard realtime correctness.

```eel
n = msg_peer_count("tempo", 1); // subscribers
id = msg_peer_id("tempo", 1, 0);
msg_peer_name(id, #name);
msg_peer_uid(id, #uid);
caps = msg_peer_caps(id);
alive = msg_peer_alive(id);
```

Roles:

- `1` = subscribers
- `2` = publishers
- `3` = either

## Shared memory

Attach a namespace:

```eel
gmem_attach("shared_env");
```

Optional size request:

```eel
gmem_attach_size("shared_env", 4096);
```

Scalar access:

```eel
x = gmem[0];
gmem[1] = 0.5;
```

Bulk access:

```eel
gmem_put(0, local_buf, len);
gmem_get(local_buf, 0, len);
```

## Messaging examples

Scalar receive:

```eel
while (msg_recv("tempo", src, tag, a, b, c, d)) (
  bpm = a;
  beatpos = b;
  playing = c;
);
```

Buffer receive:

```eel
while (msg_kind("analysis") == 2) (
  n = msg_recv_buf("analysis", src, tag, recvbuf, 512);
  n > 0 ? process = 1;
);
```

## Limits and caveats

The current implementation uses shared-memory IPC for both `gmem` namespaces and the `msg_*` peer registry/message ring. Broadcast is no-self by default. Direct sends can target any known instance ID in the same communication domain.

This is not sample-accurate inter-instance feedback. Do not use it for same-sample dependency loops.

## IPC probe plugin

`tests/dsp-jsfx-comm/ipc_probe.jsfx` is a minimal two-instance smoke test for the communication layer.

Usage:

1. Build or load two instances of the same generated plugin.
2. Leave both on the same `Bus Name` string.
3. Set one instance to `Sender` and the other to `Receiver`.
4. The receiver should show a rising `RX seq` and nonzero `RX count` in `@gfx`.

This tests both sides of the framework:

- `msg_send()` / `msg_recv()` over the shared-memory IPC ring.
- `msg_peer_count()` over the shared-memory peer registry.
- `gmem[]` over the shared-memory namespace.

Large payloads should still use `gmem_put()` / `gmem_get()`. The IPC message ring is for block-sized scalar messages and small tagged buffers.
