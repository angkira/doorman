#!/bin/bash
# iGPU FPS profiling script - Tests MIGraphX backend only
# Tests different FPS settings and measures performance + GPU load

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
FPS_VARIANTS=(30 60 120)
TEST_DURATION=30  # seconds per test
RESULTS_DIR="benchmark_results/igpu_fps_profiling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Doorman iGPU FPS Profiling (MIGraphX) ===${NC}"
echo "Backend: MIGraphX (AMD iGPU ROCm)"
echo "Test duration: ${TEST_DURATION}s per config"
echo "FPS variants: ${FPS_VARIANTS[@]}"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if radeontop is available for GPU monitoring
HAS_RADEONTOP=false
if command -v radeontop &> /dev/null; then
    HAS_RADEONTOP=true
    echo -e "${GREEN}✓ radeontop found - will monitor GPU load${NC}"
else
    echo -e "${YELLOW}⚠ radeontop not found - install with: sudo pacman -S radeontop${NC}"
fi

# Function to monitor GPU with radeontop
monitor_gpu() {
    local output_file=$1
    local duration=$2
    
    if [ "$HAS_RADEONTOP" = true ]; then
        # Run radeontop in dump mode
        timeout ${duration}s radeontop -d - -l 1 > "$output_file" 2>&1 || true
    fi
}

# Function to extract stats from daemon logs
extract_daemon_stats() {
    local log_file=$1
    local output_file=$2
    
    echo "=== Performance Statistics ===" > "$output_file"
    echo "" >> "$output_file"
    
    # Extract FPS measurements
    echo "--- Frame Capture FPS ---" >> "$output_file"
    grep "Camera capture:" "$log_file" | tail -20 >> "$output_file"
    echo "" >> "$output_file"
    
    echo "--- Detection Processing FPS ---" >> "$output_file"
    grep "Detection processing:" "$log_file" | tail -20 >> "$output_file"
    echo "" >> "$output_file"
    
    echo "--- Recognition Processing FPS ---" >> "$output_file"
    grep "Recognition processing:" "$log_file" | tail -20 >> "$output_file"
    echo "" >> "$output_file"
    
    # Calculate averages
    echo "--- Average FPS ---" >> "$output_file"
    
    capture_fps=$(grep "Camera capture:" "$log_file" | awk '{print $NF}' | sed 's/ fps//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    detection_fps=$(grep "Detection processing:" "$log_file" | awk '{print $NF}' | sed 's/ fps//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    recognition_fps=$(grep "Recognition processing:" "$log_file" | awk '{print $NF}' | sed 's/ fps//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    
    echo "Average Camera Capture: $capture_fps fps" >> "$output_file"
    echo "Average Detection: $detection_fps fps" >> "$output_file"
    echo "Average Recognition: $recognition_fps fps" >> "$output_file"
    echo "" >> "$output_file"
    
    # Extract processing times
    echo "--- Processing Times ---" >> "$output_file"
    grep "took" "$log_file" | tail -20 >> "$output_file"
}

# Function to run single test
run_test() {
    local fps=$1
    local test_name="fps_${fps}"
    local test_dir="$RESULTS_DIR/$test_name"
    mkdir -p "$test_dir"
    
    echo -e "\n${BLUE}=== Testing: ${fps} FPS ===${NC}"
    
    # Create temporary config
    local config_file="$test_dir/doorman.toml"
    cat > "$config_file" <<EOF
[daemon]
socket_path = "/run/user/$(id -u)/doorman.sock"
data_dir = "$HOME/.local/share/doorman"
log_level = "info"
processing_fps = ${fps}
user_mode = true
debug_mode = true
preview_mode = false

[camera]
device_index = 0
width = 1024
height = 720
fps = 30

[ml]
models_dir = "$HOME/.local/share/doorman/models"
backend = "migraphx"
device = "rocm"
cpu_threads = 0

[authentication]
similarity_threshold = 0.65
auth_frames = 10
timeout_secs = 3

[enrollment]
enroll_frames = 300
min_valid_frames = 50

[preprocessing]
image_width = 256
image_height = 256
filter_type = "lanczos3"
EOF
    
    # Build if needed
    if [ ! -f "target/release/doormand" ]; then
        echo "Building doormand with MIGraphX backend..."
        cargo build --release --bin doormand --features backend-migraphx
    fi
    
    # Start daemon with custom config
    echo "Starting daemon (${fps} FPS)..."
    local log_file="$test_dir/daemon.log"
    DOORMAN_CONFIG="$config_file" ./target/release/doormand --user --preview > "$log_file" 2>&1 &
    local daemon_pid=$!
    
    # Wait for daemon to initialize
    sleep 3
    
    # Check if daemon is running
    if ! kill -0 $daemon_pid 2>/dev/null; then
        echo -e "${YELLOW}⚠ Daemon failed to start${NC}"
        cat "$log_file"
        return 1
    fi
    
    echo "Daemon started (PID: $daemon_pid)"
    
    # Start GPU monitoring in background
    local gpu_log="$test_dir/gpu.log"
    if [ "$HAS_RADEONTOP" = true ]; then
        echo "Starting GPU monitoring..."
        monitor_gpu "$gpu_log" $TEST_DURATION &
        local gpu_pid=$!
    fi
    
    # Let it run for test duration
    echo "Running test for ${TEST_DURATION}s..."
    sleep $TEST_DURATION
    
    # Stop daemon
    echo "Stopping daemon..."
    kill $daemon_pid 2>/dev/null || true
    wait $daemon_pid 2>/dev/null || true
    
    # Wait for GPU monitor to finish
    if [ "$HAS_RADEONTOP" = true ]; then
        wait $gpu_pid 2>/dev/null || true
    fi
    
    # Extract statistics
    echo "Extracting statistics..."
    extract_daemon_stats "$log_file" "$test_dir/stats.txt"
    
    # Create summary
    local summary_file="$test_dir/summary.json"
    cat > "$summary_file" <<EOF
{
  "test_name": "$test_name",
  "config": {
    "processing_fps": ${fps},
    "camera_fps": 30,
    "resolution": "1024x720",
    "backend": "migraphx",
    "device": "rocm"
  },
  "test_duration": ${TEST_DURATION},
  "timestamp": "$(date -Iseconds)",
  "system": {
    "cpu": "$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)",
    "gpu": "$(lspci | grep -i vga | cut -d: -f3 | xargs)",
    "rocm_version": "$(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')"
  }
}
EOF
    
    echo -e "${GREEN}✓ Test complete: $test_name${NC}"
    cat "$test_dir/stats.txt"
}

# Run tests for each FPS variant
for fps in "${FPS_VARIANTS[@]}"; do
    run_test $fps
    sleep 2  # Cool down between tests
done

# Generate comparison report
echo -e "\n${BLUE}=== Generating Comparison Report ===${NC}"
REPORT="$RESULTS_DIR/comparison_report.md"

cat > "$REPORT" <<EOF
# Doorman iGPU FPS Profiling Results (MIGraphX Backend)

**Date:** $(date)
**System:** $(uname -a)
**CPU:** $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)
**GPU:** $(lspci | grep -i vga | cut -d: -f3 | xargs)
**ROCm:** $(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')

## Test Configuration

- Test duration: ${TEST_DURATION}s per config
- Camera resolution: 1024x720 @ 30 FPS
- Backend: MIGraphX (AMD ROCm)
- Device: iGPU (Radeon 780M)
- Processing FPS variants: ${FPS_VARIANTS[@]}

## Results

| FPS Setting | Avg Capture FPS | Avg Detection FPS | Avg Recognition FPS |
|-------------|-----------------|-------------------|---------------------|
EOF

# Add results for each test
for fps in "${FPS_VARIANTS[@]}"; do
    test_dir="$RESULTS_DIR/fps_${fps}"
    if [ -f "$test_dir/stats.txt" ]; then
        capture=$(grep "Average Camera Capture:" "$test_dir/stats.txt" | awk '{print $4}')
        detection=$(grep "Average Detection:" "$test_dir/stats.txt" | awk '{print $3}')
        recognition=$(grep "Average Recognition:" "$test_dir/stats.txt" | awk '{print $3}')
        echo "| ${fps} | ${capture:-N/A} | ${detection:-N/A} | ${recognition:-N/A} |" >> "$REPORT"
    fi
done

cat >> "$REPORT" <<EOF

## Detailed Results

EOF

# Link to detailed results
for fps in "${FPS_VARIANTS[@]}"; do
    echo "### ${fps} FPS" >> "$REPORT"
    echo "" >> "$REPORT"
    if [ -f "$RESULTS_DIR/fps_${fps}/stats.txt" ]; then
        echo "\`\`\`" >> "$REPORT"
        cat "$RESULTS_DIR/fps_${fps}/stats.txt" >> "$REPORT"
        echo "\`\`\`" >> "$REPORT"
    fi
    echo "" >> "$REPORT"
done

echo -e "${GREEN}✓ Profiling complete!${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Summary report: $REPORT"
echo ""
cat "$REPORT"
