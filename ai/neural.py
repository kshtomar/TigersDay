import os
import numpy as np
from game.constants import GAME_VECTOR_LENGTH, MOVE_VECTOR_LENGTH, MOVE_SPACE, NODES, COASTAL_INDICES

# ---------------------------------------------------------------------------
# Optional PyTorch import (for local development & training)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional ONNX Runtime import (for serverless production inference)
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ===========================================================================
# ONNX Runtime Model Wrapper (Ultra-lightweight for Vercel Serverless)
# ===========================================================================
class ONNXAlphaTiger:
    """
    Executes fast CPU inference using onnxruntime.
    Compatible with MCTS predict(state) interface.
    """
    def __init__(self, onnx_model_path: str):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed in the current environment.")
        
        # Optimize CPU execution for serverless container
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, state):
        """
        Evaluates game state vector:
        Returns:
            value (float): Scalar board evaluation in [-1.0, 1.0]
            policy_logits (np.ndarray): 1D array of logits over MOVE_VECTOR_LENGTH
        """
        x = np.asarray(state.vector, dtype=np.float32).reshape(1, -1)
        outputs = self.session.run(self.output_names, {self.input_name: x})
        
        value = float(np.squeeze(outputs[0]))
        policy_logits = np.squeeze(outputs[1], axis=0)
        return value, policy_logits


# ===========================================================================
# Dummy Fallback Model (Prevents Server Crashes if Checkpoints are Missing)
# ===========================================================================
class DummyAlphaTiger:
    """Provides uniform random/neutral priors if no model file is found."""
    def predict(self, state):
        return 0.0, np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)


# ===========================================================================
# PyTorch AlphaTiger Architecture (Used during training / local execution)
# ===========================================================================
if TORCH_AVAILABLE:
    class AlphaTiger(nn.Module):
        def __init__(self, input_size=GAME_VECTOR_LENGTH, hidden_size=256, use_factorization=True, policy_out_size=None):
            super(AlphaTiger, self).__init__()
            self.use_factorization = use_factorization

            if self.use_factorization:
                self.rn_size = NODES * len(COASTAL_INDICES)
                self.st_size = NODES * len(COASTAL_INDICES)

                target_indices = []
                offset = 0
                for name, size, move_type in MOVE_SPACE:
                    if name == "Royal Navy":
                        self.rn_start = offset
                    elif name == "Sea Trade":
                        self.st_start = offset
                    else:
                        target_indices.extend(range(offset, offset + size))
                    offset += size

                self.base_size = len(target_indices)
                self.factorized_size = len(target_indices) + NODES * 4
                self.register_buffer("base_idx_map", torch.tensor(target_indices, dtype=torch.long))

                rn_src_list, rn_dest_list = [], []
                st_src_list, st_dest_list = [], []

                for idx in range(self.rn_size):
                    node_idx = idx // len(COASTAL_INDICES)
                    coast_idx = COASTAL_INDICES[idx % len(COASTAL_INDICES)]
                    rn_src_list.append(node_idx)
                    rn_dest_list.append(coast_idx)
                    st_src_list.append(coast_idx)
                    st_dest_list.append(node_idx)
                
                self.register_buffer("rn_src_idx", torch.tensor(rn_src_list, dtype=torch.long))
                self.register_buffer("rn_dest_idx", torch.tensor(rn_dest_list, dtype=torch.long))
                self.register_buffer("st_src_idx", torch.tensor(st_src_list, dtype=torch.long))
                self.register_buffer("st_dest_idx", torch.tensor(st_dest_list, dtype=torch.long))
                
                final_policy_size = self.factorized_size
            else:
                final_policy_size = policy_out_size if policy_out_size else MOVE_VECTOR_LENGTH

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)

            self.value_fc1 = nn.Linear(hidden_size, 64)
            self.value_fc2 = nn.Linear(64, 1)

            self.policy_fc1 = nn.Linear(hidden_size, hidden_size)
            self.policy_fc2 = nn.Linear(hidden_size, final_policy_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

            value = F.relu(self.value_fc1(x))
            value = torch.tanh(self.value_fc2(value))

            policy = F.relu(self.policy_fc1(x))
            raw_logits = self.policy_fc2(policy)
            
            if not self.use_factorization:
                return value, raw_logits
            
            idx = 0
            base_logits = raw_logits[:, idx : idx + self.base_size]; idx += self.base_size
            rn_src  = raw_logits[:, idx : idx + NODES]; idx += NODES
            rn_dest = raw_logits[:, idx : idx + NODES]; idx += NODES
            st_src  = raw_logits[:, idx : idx + NODES]; idx += NODES
            st_dest = raw_logits[:, idx : idx + NODES]

            final_logits = raw_logits.new_zeros((raw_logits.shape[0], MOVE_VECTOR_LENGTH))
            final_logits[:, self.base_idx_map] = base_logits
            final_logits[:, self.rn_start : self.rn_start + self.rn_size] = rn_src[:, self.rn_src_idx] + rn_dest[:, self.rn_dest_idx]
            final_logits[:, self.st_start : self.st_start + self.st_size] = st_src[:, self.st_src_idx] + st_dest[:, self.st_dest_idx]

            return value, final_logits
        
        @torch.no_grad()
        def predict(self, state):
            device = next(self.parameters()).device
            x = torch.tensor(state.vector, dtype=torch.float32, device=device).unsqueeze(0)
            value, policy_logits = self.forward(x)
            return value.item(), policy_logits.squeeze(0).cpu().numpy()

    def save_checkpoint(model, optimizer, iteration, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
        }, path)

    def load_checkpoint(model, optimizer, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('iteration', 0)

    def load_dynamic_model(path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        uses_factorization = 'base_idx_map' in state_dict
        policy_out_size = state_dict['policy_fc2.weight'].shape[0] if 'policy_fc2.weight' in state_dict else None
            
        model = AlphaTiger(
            use_factorization=uses_factorization, 
            policy_out_size=policy_out_size
        ).to(device)
        
        model.load_state_dict(state_dict)
        return model


# ===========================================================================
# Model Loader Dispatcher (Prioritizes ONNX on serverless, fallback to Torch)
# ===========================================================================
def load_ai_model(model_path: str = None):
    """
    Dynamically loads either an ONNX model or PyTorch model based on availability.
    """
    # Candidate search paths
    candidates = []
    if model_path:
        candidates.append(model_path)
    
    # Check common ONNX model locations
    candidates.extend([
        "ai/models/alphatiger.onnx",
        "ai/models/alphatigerv13.onnx",
        "checkpoints/alphatiger.onnx",
        "public/alphatiger.onnx"
    ])
    
    # Try ONNX first (ideal for Vercel)
    if ONNX_AVAILABLE:
        for candidate in candidates:
            if candidate.endswith(".onnx") and os.path.exists(candidate):
                try:
                    print(f"✅ Loading ONNX model from {candidate}")
                    return ONNXAlphaTiger(candidate)
                except Exception as e:
                    print(f"⚠️ Failed to load ONNX model {candidate}: {e}")

    # Fallback to PyTorch (local dev)
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        try:
            device = torch.device("cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            uses_factorization = 'base_idx_map' in state_dict
            policy_out_size = state_dict['policy_fc2.weight'].shape[0] if 'policy_fc2.weight' in state_dict else None
            
            model = AlphaTiger(use_factorization=uses_factorization, policy_out_size=policy_out_size)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Loaded PyTorch model from {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load PyTorch model: {e}")

    print("⚠️ No valid ONNX or PyTorch weights found. Initializing dummy model.")
    return DummyAlphaTiger()