#!/usr/bin/env python3
"""Create a dummy liveness ONNX model for testing.

This creates a simple model that always returns 'real' for testing purposes.
NOT for production use!
"""

import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch is required to generate the dummy model")
    print("Install it with: pip install torch")
    sys.exit(1)


class DummyLivenessModel(nn.Module):
    """Minimal CNN that always returns 'real' classification."""
    
    def __init__(self):
        super().__init__()
        # Simple conv layers to match expected input/output
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 3)  # 3 classes: [real, print, replay]
        
        # Initialize weights to favor 'real' class
        with torch.no_grad():
            self.fc.weight.fill_(0.0)
            self.fc.bias[0] = 10.0  # Real class gets high score
            self.fc.bias[1] = -5.0  # Print attack gets low score
            self.fc.bias[2] = -5.0  # Replay attack gets low score
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_dummy_model(output_path: Path):
    """Create and export a dummy liveness model.
    
    Args:
        output_path: Where to save the ONNX model
    """
    print("Creating dummy liveness model...")
    
    # Create model
    model = DummyLivenessModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 80, 80)
    
    # Test it
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)
        print(f"Test output probabilities: {probs.numpy()}")
        print(f"  Real: {probs[0, 0].item():.4f}")
        print(f"  Print: {probs[0, 1].item():.4f}")
        print(f"  Replay: {probs[0, 2].item():.4f}")
    
    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\n✅ Dummy model created: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    print("\n⚠️  WARNING: This is a DUMMY model for testing only!")
    print("   It will ALWAYS classify faces as 'real'.")
    print("   DO NOT use in production!")


if __name__ == "__main__":
    # Default to data/models directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_output = project_root / "data" / "models" / "liveness.onnx"
    
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = default_output
    
    create_dummy_model(output_path)

