#!/bin/bash
export LD_LIBRARY_PATH="$HOME/.local/lib/doorman:$LD_LIBRARY_PATH"
export ORT_DYLIB_PATH="$HOME/.local/lib/doorman/libonnxruntime.so"
exec ./target/release/doormand "$@"
