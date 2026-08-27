import torch
from ai.neural import AlphaTiger
from game.constants import GAME_VECTOR_LENGTH, DEFAULT_MODEL

def export_model(tiger_path = DEFAULT_MODEL, onnx_path="ai/models/alphatiger.onnx"):
    print(f"Loading weights from {tiger_path}...")
    
    # Load with CPU mapping safely
    checkpoint = torch.load(tiger_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Dynamic architecture detection
    uses_factorization = 'base_idx_map' in state_dict
    policy_out_size = state_dict['policy_fc2.weight'].shape[0] if 'policy_fc2.weight' in state_dict else None

    # Initialize and load weights
    model = AlphaTiger(use_factorization=uses_factorization, policy_out_size=policy_out_size)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input with Float32 typing
    dummy_input = torch.randn(1, GAME_VECTOR_LENGTH, dtype=torch.float32)

    print(f"Exporting to ONNX at {onnx_path}...")
    
    # Export with dynamic axes for batching
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['value', 'policy_logits'],
        dynamic_axes={
            'board_state': {0: 'batch_size'}, 
            'value': {0: 'batch_size'}, 
            'policy_logits': {0: 'batch_size'}
        }
    )
    
    print("✅ Model successfully exported to ONNX!")

if __name__ == "__main__":
    export_model()