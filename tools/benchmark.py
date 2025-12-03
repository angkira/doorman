#!/usr/bin/env python3
"""
Unified Doorman Benchmark System
Supports multiple benchmark modes with JSON configuration
"""

import sys
import json
import time
import argparse
import os
import subprocess
import base64
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import io

# Setup environment for AMD Radeon 780M iGPU
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.1'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['GPU_MAX_HW_QUEUES'] = '1'
os.environ['ORT_LOG_LEVEL'] = '3'

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    name: str
    mode: str  # "detection", "liveness", "embedding", "full_pipeline"
    backend: str  # "torch-direct", "torch-ipc", "tract", "ort"
    iterations: int = 100
    warmup_iterations: int = 10
    image_width: int = 1024
    image_height: int = 720
    device: str = "cuda"
    models_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class BenchmarkResult:
    """Benchmark results"""
    config: BenchmarkConfig
    mean_time_ms: float
    std_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    mean_fps: float
    median_fps: float
    min_fps: float
    max_fps: float
    first_10_avg_ms: float
    last_10_avg_ms: float
    degradation_percent: float
    stable: bool
    times: List[float]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (handle numpy types)"""
        return {
            'config': {
                'name': str(self.config.name),
                'mode': str(self.config.mode),
                'backend': str(self.config.backend),
                'iterations': int(self.config.iterations),
                'warmup_iterations': int(self.config.warmup_iterations),
                'image_width': int(self.config.image_width),
                'image_height': int(self.config.image_height),
                'device': str(self.config.device),
            },
            'mean_time_ms': float(self.mean_time_ms),
            'std_time_ms': float(self.std_time_ms),
            'median_time_ms': float(self.median_time_ms),
            'min_time_ms': float(self.min_time_ms),
            'max_time_ms': float(self.max_time_ms),
            'mean_fps': float(self.mean_fps),
            'median_fps': float(self.median_fps),
            'min_fps': float(self.min_fps),
            'max_fps': float(self.max_fps),
            'first_10_avg_ms': float(self.first_10_avg_ms),
            'last_10_avg_ms': float(self.last_10_avg_ms),
            'degradation_percent': float(self.degradation_percent),
            'stable': bool(self.stable),
            'sample_count': int(len(self.times)),
        }

class TorchDirectBackend:
    """Direct Python inference (no IPC)"""

    def __init__(self, models_dir: str, device: str = "cuda"):
        sys.path.insert(0, str(Path(__file__).parent.parent / 'daemon' / 'src' / 'ml'))
        from torch_inference import TorchInferenceBackend

        self.backend = TorchInferenceBackend(models_dir, device)
        print(f"[TorchDirect] Initialized on {device}")

    def detect_faces(self, image_data: bytes, width: int, height: int) -> dict:
        return self.backend.detect_faces(image_data, width, height)

    def check_liveness(self, face_crop: bytes) -> dict:
        return self.backend.check_liveness(face_crop)

    def extract_embedding(self, face_crop: bytes) -> dict:
        return self.backend.extract_embedding(face_crop)

class TorchIPCBackend:
    """PyTorch inference via IPC subprocess (like Rust daemon)"""

    def __init__(self, models_dir: str, device: str = "cuda"):
        self.models_dir = models_dir
        self.device = device
        self.request_id = 0

        # Find Python script
        script_path = Path(__file__).parent.parent / 'daemon' / 'src' / 'ml' / 'torch_inference.py'
        if not script_path.exists():
            raise FileNotFoundError(f"torch_inference.py not found: {script_path}")

        # Start Python subprocess
        print(f"[TorchIPC] Starting subprocess: {script_path}")
        venv_python = os.environ.get('VIRTUAL_ENV', '')
        if venv_python:
            python_cmd = f"{venv_python}/bin/python3"
        else:
            python_cmd = "python3"

        self.process = subprocess.Popen(
            [python_cmd, str(script_path), models_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        print(f"[TorchIPC] Subprocess started (PID: {self.process.pid})")

    def _call_method(self, method: str, params: dict) -> dict:
        """Call JSON-RPC method"""
        self.request_id += 1

        request = {
            "id": self.request_id,
            "method": method,
            "params": params
        }

        # Send request
        request_json = json.dumps(request)
        self.process.stdin.write(request_json + '\n')
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from subprocess")

        response = json.loads(response_line)

        if 'error' in response:
            raise RuntimeError(f"IPC error: {response['error']}")

        return response.get('result', {})

    def detect_faces(self, image_data: bytes, width: int, height: int) -> dict:
        """Detect faces via IPC"""
        # Encode image to base64 (like Rust does)
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        params = {
            "image_data": image_b64,
            "width": width,
            "height": height
        }

        return self._call_method("detect_faces", params)

    def check_liveness(self, face_crop: bytes) -> dict:
        """Check liveness via IPC"""
        face_b64 = base64.b64encode(face_crop).decode('utf-8')
        params = {"face_crop": face_b64}
        return self._call_method("check_liveness", params)

    def extract_embedding(self, face_crop: bytes) -> dict:
        """Extract embedding via IPC"""
        face_b64 = base64.b64encode(face_crop).decode('utf-8')
        params = {"face_crop": face_b64}
        return self._call_method("extract_embedding", params)

    def __del__(self):
        """Clean up subprocess"""
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
            self.process.wait(timeout=5)

class TorchNativeBackend:
    """Native PyO3 extension (zero IPC overhead)"""

    def __init__(self, models_dir: str, device: str = "cuda"):
        try:
            from doorman_ml import DoormanML
        except ImportError:
            raise RuntimeError(
                "doorman_ml_native not installed. Build it first:\n"
                "  cd daemon/native_ml && ./build.sh"
            )

        self.ml = DoormanML(models_dir, device)
        print(f"[TorchNative] Initialized native extension on {device}")

    def detect_faces(self, image_data: bytes, width: int, height: int) -> dict:
        """Detect faces via native extension"""
        # Convert JPEG to RGB bytes
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb_bytes = img.tobytes()

        # Call native function
        detections = self.ml.detect_faces(rgb_bytes, width, height)
        
        # Convert to dict format
        return {
            "detections": [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "landmarks": det.landmarks
                }
                for det in detections
            ]
        }

    def check_liveness(self, face_crop: bytes) -> dict:
        """Check liveness via native extension"""
        img = Image.open(io.BytesIO(face_crop))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb_bytes = img.tobytes()

        result = self.ml.check_liveness(rgb_bytes)
        return {
            "is_live": result.is_live,
            "confidence": result.confidence
        }

    def extract_embedding(self, face_crop: bytes) -> dict:
        """Extract embedding via native extension"""
        img = Image.open(io.BytesIO(face_crop))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb_bytes = img.tobytes()

        embedding_bytes = self.ml.extract_embedding(rgb_bytes)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        return {
            "embedding": embedding.tolist()
        }

class BenchmarkRunner:
    """Main benchmark runner"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.backend = None

        # Determine models directory
        if config.models_dir:
            self.models_dir = Path(config.models_dir)
        else:
            self.models_dir = Path.home() / ".local/share/doorman/models"

        # Initialize backend
        self._init_backend()

        # Create test image
        self.test_image = self._create_test_image()

    def _init_backend(self):
        """Initialize the appropriate backend"""
        if self.config.backend == "torch-direct":
            self.backend = TorchDirectBackend(str(self.models_dir), self.config.device)
        elif self.config.backend == "torch-ipc":
            self.backend = TorchIPCBackend(str(self.models_dir), self.config.device)
        elif self.config.backend == "torch-native":
            self.backend = TorchNativeBackend(str(self.models_dir), self.config.device)
        else:
            raise ValueError(f"Backend not implemented: {self.config.backend}")

    def _create_test_image(self) -> bytes:
        """Create test image"""
        img = Image.new('RGB', (self.config.image_width, self.config.image_height),
                       color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        return buf.getvalue()

    def _run_iteration(self) -> float:
        """Run single benchmark iteration, return time in ms"""
        start = time.perf_counter()

        if self.config.mode == "detection":
            result = self.backend.detect_faces(
                self.test_image,
                self.config.image_width,
                self.config.image_height
            )
        elif self.config.mode == "full_pipeline":
            # Detection + liveness + embedding
            detection = self.backend.detect_faces(
                self.test_image,
                self.config.image_width,
                self.config.image_height
            )
            # TODO: Implement full pipeline when needed
            result = detection
        else:
            raise ValueError(f"Mode not implemented: {self.config.mode}")

        end = time.perf_counter()
        return (end - start) * 1000

    def warmup(self):
        """Warmup iterations"""
        print(f"\nWarming up ({self.config.warmup_iterations} iterations)...")
        for i in range(self.config.warmup_iterations):
            self._run_iteration()
            print(f"  Warmup {i+1}/{self.config.warmup_iterations}", end='\r')
        print()
        print("✓ Warmup complete\n")

    def run(self) -> BenchmarkResult:
        """Run benchmark and return results"""
        print(f"Running benchmark: {self.config.name}")
        print(f"  Mode: {self.config.mode}")
        print(f"  Backend: {self.config.backend}")
        print(f"  Iterations: {self.config.iterations}")
        print(f"  Image: {self.config.image_width}x{self.config.image_height}")

        # Warmup
        self.warmup()

        # Benchmark
        print(f"Running {self.config.iterations} iterations...\n")
        times = []

        for i in range(self.config.iterations):
            elapsed_ms = self._run_iteration()
            times.append(elapsed_ms)

            if (i+1) % 10 == 0:
                current_avg = np.mean(times[-10:])
                current_fps = 1000 / current_avg
                print(f"  Progress: {i+1}/{self.config.iterations} | "
                      f"Last 10 avg: {current_avg:.1f}ms ({current_fps:.1f} FPS)")

        # Calculate statistics
        times_array = np.array(times)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        median_time = np.median(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)

        mean_fps = 1000 / mean_time
        median_fps = 1000 / median_time
        min_fps = 1000 / max_time
        max_fps = 1000 / min_time

        # Check degradation
        first_10 = np.mean(times[:10])
        last_10 = np.mean(times[-10:])
        degradation = ((last_10 - first_10) / first_10) * 100

        stable = abs(degradation) < 5

        return BenchmarkResult(
            config=self.config,
            mean_time_ms=float(mean_time),
            std_time_ms=float(std_time),
            median_time_ms=float(median_time),
            min_time_ms=float(min_time),
            max_time_ms=float(max_time),
            mean_fps=float(mean_fps),
            median_fps=float(median_fps),
            min_fps=float(min_fps),
            max_fps=float(max_fps),
            first_10_avg_ms=float(first_10),
            last_10_avg_ms=float(last_10),
            degradation_percent=float(degradation),
            stable=stable,
            times=times
        )

def print_results(result: BenchmarkResult):
    """Print benchmark results"""
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print()

    print(f"Inference Time (ms):")
    print(f"  Mean:   {result.mean_time_ms:.2f} ± {result.std_time_ms:.2f}")
    print(f"  Median: {result.median_time_ms:.2f}")
    print(f"  Min:    {result.min_time_ms:.2f}")
    print(f"  Max:    {result.max_time_ms:.2f}")
    print()

    print(f"Throughput (FPS):")
    print(f"  Mean:   {result.mean_fps:.2f}")
    print(f"  Median: {result.median_fps:.2f}")
    print(f"  Min:    {result.min_fps:.2f}")
    print(f"  Max:    {result.max_fps:.2f}")
    print()

    print(f"Performance Stability:")
    print(f"  First 10 avg:  {result.first_10_avg_ms:.2f}ms ({1000/result.first_10_avg_ms:.1f} FPS)")
    print(f"  Last 10 avg:   {result.last_10_avg_ms:.2f}ms ({1000/result.last_10_avg_ms:.1f} FPS)")
    print(f"  Degradation:   {result.degradation_percent:+.1f}%")
    print()

    if abs(result.degradation_percent) > 20:
        print("⚠️  WARNING: Significant performance degradation detected!")
    elif result.degradation_percent > 5:
        print("⚠️  Performance slightly degraded over time")
    elif result.degradation_percent < -5:
        print("✓ Performance improved over time (JIT warmup)")
    else:
        print("✓ Performance stable")

    print()
    print("="*60)

def save_results(result: BenchmarkResult, output_dir: Path):
    """Save benchmark results to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{result.config.backend}_{result.config.mode}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path

def load_config_file(config_path: Path) -> List[BenchmarkConfig]:
    """Load benchmark configurations from JSON file"""
    with open(config_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [BenchmarkConfig.from_dict(item) for item in data]
    else:
        return [BenchmarkConfig.from_dict(data)]

def main():
    parser = argparse.ArgumentParser(description="Unified Doorman Benchmark")
    parser.add_argument('-c', '--config', type=Path,
                       help='JSON configuration file')
    parser.add_argument('-o', '--output', type=Path,
                       default=Path('benchmark_results'),
                       help='Output directory for results (default: benchmark_results)')

    # Quick run options (if no config file)
    parser.add_argument('--mode', choices=['detection', 'liveness', 'embedding', 'full_pipeline'],
                       default='detection', help='Benchmark mode')
    parser.add_argument('--backend', choices=['torch-direct', 'torch-ipc', 'tract', 'ort'],
                       default='torch-direct', help='Backend to use')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup iterations')

    args = parser.parse_args()

    # Load configurations
    if args.config:
        print(f"Loading configuration from: {args.config}")
        configs = load_config_file(args.config)
    else:
        # Create single config from CLI args
        configs = [BenchmarkConfig(
            name=f"{args.backend}_{args.mode}",
            mode=args.mode,
            backend=args.backend,
            iterations=args.iterations,
            warmup_iterations=args.warmup
        )]

    print(f"\n{'='*60}")
    print("Doorman Unified Benchmark System")
    print(f"{'='*60}")
    print(f"Running {len(configs)} benchmark(s)\n")

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config.name}")
        print("-" * 60)

        runner = BenchmarkRunner(config)
        result = runner.run()

        print_results(result)
        save_results(result, args.output)

        results.append(result)

    print(f"\n{'='*60}")
    print("All benchmarks complete!")
    print(f"{'='*60}")
    print(f"\nResults directory: {args.output}")

if __name__ == "__main__":
    main()
