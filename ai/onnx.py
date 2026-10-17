import argparse
import torch
from ai.neural import AlphaTiger
from game.constants import GAME_VECTOR_LENGTH


def export_model(tiger_path: str, onnx_path: str):
    print(f"Loading weights from {tiger_path}...")

    # Use the latest architecture
    model = AlphaTiger()
    checkpoint = torch.load(tiger_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    dummy_input = torch.randn(1, GAME_VECTOR_LENGTH, dtype=torch.float32)
    
    # Export with dynamic axes for batching
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["board_state"],
        output_names=["value", "policy_logits"],
        dynamic_axes={
            "board_state": {0: "batch_size"},
            "value": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
        },
    )
    print(f"✅ Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch AlphaTiger model to ONNX")
    parser.add_argument("--input", default="checkpoints/final.pt", help="Path to input .pt checkpoint")
    parser.add_argument("--output", default="ai/models/alphatiger.onnx", help="Path to output .onnx file")
    args = parser.parse_args()

    export_model(tiger_path=args.input, onnx_path=args.output)