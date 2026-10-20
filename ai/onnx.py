import argparse
from pathlib import Path
import torch
from ai.neural import AlphaTiger, MANIFEST, load_checkpoint
from game.constants import GAME_VECTOR_LENGTH


def export_model(tiger_path: str, onnx_path: str):
    print(f"Loading weights from {tiger_path}...")

    # Use the latest architecture
    config = MANIFEST.get(Path(tiger_path).stem, {})
    model = AlphaTiger(use_factorization=config.get('factorized', True))
    load_checkpoint(model, None, tiger_path)
    model.eval()

    dummy_input = torch.randn(2, GAME_VECTOR_LENGTH, dtype=torch.float32)
    
    # Export with dynamic axes for batching
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        dynamic_shapes=({0: torch.export.Dim("batch_size")},),
        external_data=False,
    )
    print(f"✅ Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch AlphaTiger model to ONNX")
    parser.add_argument("--input", default="checkpoints/final.pt", help="Path to input .pt checkpoint")
    parser.add_argument("--output", default="ai/models/alphatiger.onnx", help="Path to output .onnx file")
    args = parser.parse_args()

    export_model(tiger_path=args.input, onnx_path=args.output)
