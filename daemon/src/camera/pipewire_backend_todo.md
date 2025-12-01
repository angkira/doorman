# PipeWire Backend Implementation TODO

## Summary

GStreamer с pipewiresrc **НЕ РАБОТАЕТ** - не находит камеру даже после освобождения.

**Решение**: Использовать `pipewire` Rust crate напрямую через `pw::stream::Stream` API.

## Implementation Plan

### 1. Add dependencies (daemon/Cargo.toml)

```toml
[dependencies]
pipewire = { version = "0.8", optional = true }
libspa = { version = "0.8", optional = true }
libspa-sys = { version = "0.8", optional = true }

[features]
camera-pipewire = ["pipewire", "libspa", "libspa-sys"]
```

### 2. Create backend (daemon/src/camera/pipewire_backend.rs)

Key components:
- `PipeWireCamera` struct with MainLoop and Stream
- Frame buffer channel (mpsc)
- Event listener for `process` callback
- Format negotiation (RGB, resolution, fps)

### 3. Integration

Update `daemon/src/camera/mod.rs`:
```rust
#[cfg(feature = "camera-pipewire")]
mod pipewire_backend;

#[cfg(feature = "camera-pipewire")]
pub use pipewire_backend::PipeWireCamera;
```

### 4. Test

```bash
cargo build --release --features camera-pipewire
./target/release/doormand --user --preview
```

## Status

- [ ] Add dependencies
- [ ] Implement PipeWireCamera
- [ ] Integrate with Camera enum
- [ ] Test capture
- [ ] Benchmark performance

**ETA**: 30-60 minutes for basic implementation

## Alternative: Just use FFmpeg

FFmpeg **РАБОТАЕТ СЕЙЧАС**! Просто медленно (10 fps). Это приемлемо для MVP.

Decision: 
- **Short term**: Use FFmpeg (works now)
- **Long term**: Implement pw-stream backend (proper solution)
