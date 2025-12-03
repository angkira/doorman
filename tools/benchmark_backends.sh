#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/benchmark_$TIMESTAMP.json"

mkdir -p "$RESULTS_DIR"

# System info collection
get_system_info() {
    cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "cpu": "$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)",
    "cpu_cores": $(nproc),
    "memory_gb": $(free -g | awk '/^Mem:/{print $2}'),
    "gpu": "$(lspci | grep -i 'vga\|3d\|display' | head -1 | cut -d: -f3 | xargs || echo 'none')"
}
EOF
}

# Build configuration
CONFIGS=(
    "tract-cpu:--features backend-tract:cpu"
    "ort-cpu:--features backend-ort-cpu:cpu"
    "ort-rocm:--features backend-ort-rocm:rocm"
)

echo "=== Doorman Backend Benchmark Suite ==="
echo "Results will be saved to: $RESULT_FILE"
echo ""

# Start JSON output
echo "{" > "$RESULT_FILE"
echo "  \"system_info\": $(get_system_info)," >> "$RESULT_FILE"
echo "  \"benchmarks\": [" >> "$RESULT_FILE"

first=true

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name features device <<< "$config"
    
    echo "----------------------------------------"
    echo "Building and testing: $name"
    echo "Features: $features"
    echo "Device: $device"
    echo "----------------------------------------"
    
    # Build
    echo "Building..."
    if ! cargo build --release $features 2>&1 | grep -E "(Compiling|Finished|error)"; then
        echo "Build failed for $name, skipping..."
        continue
    fi
    
    # Update config for this backend
    backend_type=$(echo $name | cut -d- -f1)
    
    cp doorman.toml doorman.toml.backup
    sed -i "s/backend = .*/backend = \"$backend_type\"/" doorman.toml
    sed -i "s/device = .*/device = \"$device\"/" doorman.toml
    
    # Kill any running daemon
    pkill -f doormand || true
    sleep 2
    
    # Start daemon in background
    echo "Starting daemon..."
    ./target/release/doormand --user --preview > "/tmp/doormand_$name.log" 2>&1 &
    DAEMON_PID=$!
    sleep 3
    
    # Check if daemon started
    if ! kill -0 $DAEMON_PID 2>/dev/null; then
        echo "Daemon failed to start for $name"
        cat "/tmp/doormand_$name.log"
        mv doorman.toml.backup doorman.toml
        continue
    fi
    
    # Run benchmark - use test video if available, otherwise skip
    if [ -f "2025-11-26-115723.webm" ]; then
        echo "Running benchmark (30 seconds)..."
        
        # Collect metrics
        start_time=$(date +%s.%N)
        frames_processed=0
        detections=0
        recognitions=0
        
        # Monitor logs for 30 seconds
        timeout 30s tail -f "/tmp/doormand_$name.log" 2>/dev/null | while read line; do
            if echo "$line" | grep -q "Broadcasting detection"; then
                ((detections++)) || true
            fi
            if echo "$line" | grep -q "Recognition result"; then
                ((recognitions++)) || true
            fi
            if echo "$line" | grep -q "Camera capture:.*fps"; then
                frames_processed=$(echo "$line" | grep -oP '\d+\.\d+(?= fps)' || echo "0")
            fi
        done &
        MONITOR_PID=$!
        
        # Let it run
        sleep 30
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        
        # Get resource usage
        cpu_usage=$(ps -p $DAEMON_PID -o %cpu= | xargs || echo "0")
        mem_usage=$(ps -p $DAEMON_PID -o rss= | xargs || echo "0")
        mem_mb=$(echo "scale=2; $mem_usage / 1024" | bc)
        
        # Stop monitoring
        kill $MONITOR_PID 2>/dev/null || true
        
        # Calculate averages from logs
        avg_fps=$(grep "Camera capture:" "/tmp/doormand_$name.log" | grep -oP '\d+\.\d+(?= fps)' | awk '{s+=$1; c++} END {if(c>0) print s/c; else print 0}')
        detection_fps=$(grep "Detection processing:" "/tmp/doormand_$name.log" | grep -oP '\d+\.\d+(?= fps)' | awk '{s+=$1; c++} END {if(c>0) print s/c; else print 0}')
        
        # Add comma if not first entry
        if [ "$first" = false ]; then
            echo "," >> "$RESULT_FILE"
        fi
        first=false
        
        # Write benchmark result
            # Read actual config from toml
        actual_backend=$(grep '^backend = ' doorman.toml | cut -d'"' -f2)
        actual_device=$(grep '^device = ' doorman.toml | cut -d'"' -f2)
        actual_max_fps=$(grep '^max_fps = ' doorman.toml | cut -d'=' -f2 | xargs)
        
        cat <<EOF >> "$RESULT_FILE"
    {
      "name": "$name",
      "backend": "$backend_type",
      "device": "$device",
      "features": "$features",
      "config": {
        "backend": "$actual_backend",
        "device": "$actual_device",
        "max_fps": ${actual_max_fps:-30}
      },
      "metrics": {
        "duration_seconds": $duration,
        "avg_capture_fps": ${avg_fps:-0},
        "avg_detection_fps": ${detection_fps:-0},
        "cpu_percent": ${cpu_usage:-0},
        "memory_mb": ${mem_mb:-0}
      },
      "build_info": {
        "rustc_version": "$(rustc --version)",
        "target": "$(rustc -vV | grep host | cut -d: -f2 | xargs)"
      }
    }
EOF
        
        echo ""
        echo "Results for $name:"
        echo "  Capture FPS: ${avg_fps:-0}"
        echo "  Detection FPS: ${detection_fps:-0}"
        echo "  CPU Usage: ${cpu_usage:-0}%"
        echo "  Memory: ${mem_mb:-0} MB"
        echo ""
    else
        echo "No test video found, skipping runtime benchmark"
        
        if [ "$first" = false ]; then
            echo "," >> "$RESULT_FILE"
        fi
        first=false
        
        # Read actual config from toml
        actual_backend=$(grep '^backend = ' doorman.toml | cut -d'"' -f2)
        actual_device=$(grep '^device = ' doorman.toml | cut -d'=' -f2 | xargs)
        actual_max_fps=$(grep '^max_fps = ' doorman.toml | cut -d'=' -f2 | xargs)
        
        cat <<EOF >> "$RESULT_FILE"
    {
      "name": "$name",
      "backend": "$backend_type",
      "device": "$device",
      "features": "$features",
      "config": {
        "backend": "$actual_backend",
        "device": "$actual_device",
        "max_fps": ${actual_max_fps:-30}
      },
      "metrics": {
        "note": "No test video available"
      }
    }
EOF
    fi
    
    # Cleanup
    kill $DAEMON_PID 2>/dev/null || true
    sleep 1
    mv doorman.toml.backup doorman.toml
done

# Close JSON
echo "" >> "$RESULT_FILE"
echo "  ]" >> "$RESULT_FILE"
echo "}" >> "$RESULT_FILE"

echo "========================================"
echo "Benchmark complete!"
echo "Results saved to: $RESULT_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULT_FILE | jq ."
echo ""
echo "To compare all benchmarks:"
echo "  python3 -c \"
import json
import glob
from tabulate import tabulate

files = sorted(glob.glob('$RESULTS_DIR/benchmark_*.json'))
data = []
for f in files:
    with open(f) as fp:
        result = json.load(fp)
        for bench in result['benchmarks']:
            if 'metrics' in bench and 'avg_capture_fps' in bench['metrics']:
                data.append([
                    result['system_info']['timestamp'][:10],
                    bench['name'],
                    f\\\"{bench['metrics']['avg_capture_fps']:.1f}\\\",
                    f\\\"{bench['metrics']['avg_detection_fps']:.1f}\\\",
                    f\\\"{bench['metrics']['cpu_percent']:.1f}\\\",
                    f\\\"{bench['metrics']['memory_mb']:.1f}\\\"
                ])

print(tabulate(data, headers=['Date', 'Backend', 'Capture FPS', 'Detect FPS', 'CPU%', 'Memory MB']))
\""

chmod +x "$0"
