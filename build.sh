#!/bin/bash
export LIBTORCH=$(pwd)/.venv/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export LIBTORCH_CXX11_ABI=1
cargo build --release --features backend-tch,camera-gstreamer
