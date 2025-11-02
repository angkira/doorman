"""Detailed specifications for ONNX models used in doorman."""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class TensorSpec:
    """Specification for a tensor (input or output)."""
    name: str
    shape: Tuple[int, ...]  # Use -1 for dynamic dimensions
    dtype: str  # e.g., 'float32', 'int64'
    description: str


@dataclass
class PreprocessingSpec:
    """Preprocessing requirements for a model."""
    resize: Tuple[int, int]  # (width, height)
    color_space: str  # 'RGB', 'BGR', 'GRAY'
    normalization: str  # 'divide_255', 'mean_std', 'neg1_to_1'
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    layout: str = 'NCHW'  # or 'NHWC'


@dataclass
class ModelSpec:
    """Complete specification for an ONNX model."""
    model_key: str
    name: str
    description: str
    framework: str  # 'pytorch', 'tensorflow', 'onnx', etc.
    
    # Model file
    filename: str
    url: str
    size_mb: float
    sha256: str
    
    # Input/Output specifications
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    
    # Preprocessing
    preprocessing: PreprocessingSpec
    
    # Usage notes
    notes: str = ""


# ============================================================================
# Model Specifications Registry
# ============================================================================

ULTRAFACE_RFB_320 = ModelSpec(
    model_key="blazeface",
    name="UltraFace RFB-320",
    description="Lightweight face detection model based on RFB architecture",
    framework="pytorch->onnx",
    filename="blazeface.onnx",
    url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
    size_mb=1.2,
    sha256="",
    inputs=[
        TensorSpec(
            name="input",
            shape=(1, 3, -1, -1),  # Dynamic height/width
            dtype="float32",
            description="RGB image tensor [batch, channels, height, width]"
        )
    ],
    outputs=[
        TensorSpec(
            name="boxes",
            shape=(1, -1, 4),  # Dynamic number of detections
            dtype="float32",
            description="Bounding boxes [batch, num_boxes, 4] in format [x1, y1, x2, y2]"
        ),
        TensorSpec(
            name="scores",
            shape=(1, -1, 2),  # Background + face scores
            dtype="float32",
            description="Classification scores [batch, num_boxes, 2]"
        )
    ],
    preprocessing=PreprocessingSpec(
        resize=(320, 240),  # Or any size, model is flexible
        color_space='RGB',
        normalization='divide_255',
        layout='NCHW'
    ),
    notes="""
    UltraFace is designed for edge devices with minimal memory footprint.
    - Input can be any size, but 320x240 is optimal
    - Outputs raw boxes that need NMS (non-maximum suppression)
    - Confidence threshold typically 0.5-0.7
    """
)

LIVENESS_DETECTION = ModelSpec(
    model_key="liveness",
    name="Face Liveness Detection",
    description="Anti-spoofing model for presentation attack detection",
    framework="pytorch->onnx",
    filename="liveness.onnx",
    url="",  # Manual installation required
    size_mb=4.0,
    sha256="",
    inputs=[
        TensorSpec(
            name="input",
            shape=(1, 3, 112, 112),
            dtype="float32",
            description="RGB face crop [batch, channels, height, width] - adjust based on actual model"
        )
    ],
    outputs=[
        TensorSpec(
            name="output",
            shape=(1, 2),  # or (1, 1) depending on model
            dtype="float32",
            description="Classification output - [fake, real] or single score"
        )
    ],
    preprocessing=PreprocessingSpec(
        resize=(112, 112),
        color_space='RGB',
        normalization='mean_std',
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        layout='NCHW'
    ),
    notes="""
    ⭐ RECOMMENDED: InsightFace Buffalo Models (18k+ GitHub stars)
    
    InsightFace is THE industry-standard face toolkit - production-proven and widely used.
    
    Quick Setup:
    ```bash
    pip install insightface onnxruntime
    python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"
    # Models auto-download to: ~/.insightface/models/buffalo_l/
    # Copy liveness model to doorman: data/models/liveness.onnx
    ```
    
    Alternative: MiniFASNet V2 (Silent-Face-Anti-Spoofing, 3.5k+ stars)
    - Input: (1, 3, 80, 80) RGB
    - Output: (1, 3) [real, print, replay]
    - Normalization: divide by 255
    - Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    
    For detailed instructions see: OBTAINING_LIVENESS_MODEL.md
    
    ⚠️ Note: Input/output shapes vary by model. Check your specific model's requirements.
    """
)

ARCFACE_RESNET100 = ModelSpec(
    model_key="mobilefacenet",
    name="ArcFace ResNet100",
    description="Face recognition model generating 512-dimensional embeddings",
    framework="mxnet->onnx",
    filename="mobilefacenet.onnx",
    url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
    size_mb=249.0,
    sha256="",
    inputs=[
        TensorSpec(
            name="data",
            shape=(1, 3, 112, 112),
            dtype="float32",
            description="Aligned face image [batch, channels, height, width]"
        )
    ],
    outputs=[
        TensorSpec(
            name="fc1",
            shape=(1, 512),
            dtype="float32",
            description="Face embedding vector [batch, embedding_dim]"
        )
    ],
    preprocessing=PreprocessingSpec(
        resize=(112, 112),
        color_space='RGB',
        normalization='neg1_to_1',  # (x / 127.5) - 1.0
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        layout='NCHW'
    ),
    notes="""
    ArcFace ResNet100 is a high-accuracy face recognition model.
    - Input must be aligned 112x112 face crop
    - Normalize to [-1, 1] range: (pixel / 127.5) - 1.0
    - Output embedding should be L2-normalized before comparison
    - Cosine similarity threshold typically 0.3-0.5 for matching
    
    Alternatives (also 112x112):
    - MobileFaceNet (smaller, faster): ~4MB
    - ArcFace ResNet50: ~166MB
    - ArcFace MobileFaceNet: ~3.5MB
    """
)


# Registry for easy lookup
MODEL_SPECS = {
    "blazeface": ULTRAFACE_RFB_320,
    "liveness": LIVENESS_DETECTION,
    "mobilefacenet": ARCFACE_RESNET100,
}


def get_model_spec(model_key: str) -> ModelSpec:
    """Get detailed specification for a model.
    
    Args:
        model_key: Model identifier
        
    Returns:
        ModelSpec object with complete model information
        
    Raises:
        KeyError: If model_key not found
    """
    return MODEL_SPECS[model_key]


def print_model_spec(model_key: str) -> str:
    """Generate a formatted string with model specifications.
    
    Args:
        model_key: Model identifier
        
    Returns:
        Formatted specification string
    """
    spec = get_model_spec(model_key)
    
    lines = [
        f"Model: {spec.name}",
        f"File: {spec.filename} ({spec.size_mb:.1f} MB)",
        f"Description: {spec.description}",
        f"Framework: {spec.framework}",
        "",
        "Inputs:",
    ]
    
    for inp in spec.inputs:
        lines.append(f"  - {inp.name}: {inp.shape} ({inp.dtype})")
        lines.append(f"    {inp.description}")
    
    lines.append("")
    lines.append("Outputs:")
    
    for out in spec.outputs:
        lines.append(f"  - {out.name}: {out.shape} ({out.dtype})")
        lines.append(f"    {out.description}")
    
    lines.append("")
    lines.append("Preprocessing:")
    lines.append(f"  - Resize: {spec.preprocessing.resize}")
    lines.append(f"  - Color: {spec.preprocessing.color_space}")
    lines.append(f"  - Normalization: {spec.preprocessing.normalization}")
    lines.append(f"  - Layout: {spec.preprocessing.layout}")
    
    if spec.notes:
        lines.append("")
        lines.append("Notes:")
        for line in spec.notes.strip().split("\n"):
            lines.append(f"  {line.strip()}")
    
    return "\n".join(lines)

