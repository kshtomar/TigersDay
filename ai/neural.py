import os
import json
from pathlib import Path
import numpy as np
from game.constants import *

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "models", "manifest.json")
MANIFEST = json.load(open(MANIFEST_PATH)) if os.path.exists(MANIFEST_PATH) else {}

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ONNXAlphaTiger:
    def __init__(self, onnx_model_path: str = DEFAULT_MODEL):
        # Optimize CPU execution for serverless container
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, state):
        x = np.asarray(state.vector, dtype=np.float32).reshape(1, -1)
        outputs = self.session.run(self.output_names, {self.input_name: x})
        
        value = float(np.squeeze(outputs[0]))
        policy_logits = np.squeeze(outputs[1], axis=0)
        return value, policy_logits


Module = nn.Module if nn else object


class AlphaTiger(Module):
    def __init__(self, input_size=GAME_VECTOR_LENGTH, hidden_size=256, use_factorization=True, policy_out_size=None, activation='silu'):
        super().__init__()
        self.use_factorization = use_factorization
        self.set_activation(activation)

        if self.use_factorization:
            target_indices = []
            offset = 0
            for name, size, _ in MOVE_SPACE:
                if name == "Royal Navy":
                    rn = slice(offset, offset + size)
                elif name == "Sea Trade":
                    st = slice(offset, offset + size)
                else:
                    target_indices.extend(range(offset, offset + size))
                offset += size

            base_size = len(target_indices)
            final_policy_size = base_size + NODES * 4
            nodes = torch.arange(NODES).repeat_interleave(len(COASTAL_INDICES))
            coasts = torch.tensor(COASTAL_INDICES, dtype=torch.long).repeat(NODES)
            src = torch.empty(MOVE_VECTOR_LENGTH, dtype=torch.long)
            dest = torch.full((MOVE_VECTOR_LENGTH,), final_policy_size, dtype=torch.long)
            src[target_indices] = torch.arange(base_size)
            # index map does: base + 0 or source + dest for factorization
            src[rn] = base_size + nodes
            dest[rn] = base_size + NODES + coasts
            src[st] = base_size + 2*NODES + coasts
            dest[st] = base_size + 3*NODES + nodes

            # buffer everything to gpu
            self.register_buffer("policy_src_idx", src, persistent=False)
            self.register_buffer("policy_dest_idx", dest, persistent=False)
        else:
            final_policy_size = policy_out_size if policy_out_size else MOVE_VECTOR_LENGTH

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.value_fc1 = nn.Linear(hidden_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self.policy_fc1 = nn.Linear(hidden_size, hidden_size)
        # return factorized output
        self.policy_fc2 = nn.Linear(hidden_size, final_policy_size)

    def set_activation(self, activation):
        if activation not in ('relu', 'silu'):
            raise ValueError(f"Unsupported activation: {activation!r}")
        self.activation_name = activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.SiLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        value = self.activation(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))

        policy = self.activation(self.policy_fc1(x))
        raw_logits = self.policy_fc2(policy)

        if not self.use_factorization:
            return value, raw_logits

        # unfactorize RN and ST for the game engine
        padded = torch.cat((raw_logits, raw_logits.new_zeros((raw_logits.shape[0], 1))), dim=1)
        final_logits = padded.index_select(1, self.policy_src_idx) + padded.index_select(1, self.policy_dest_idx)

        # return evaluation and policy logits
        return value, final_logits

    @torch.inference_mode()
    def predict(self, state):
        device = self.fc1.weight.device
        x = torch.tensor(state.vector, dtype=torch.float32, device=device).unsqueeze(0)
        value, policy_logits = self.forward(x)
        return value.item(), policy_logits.squeeze(0).cpu().numpy()


def save_checkpoint(model, optimizer, iteration, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'model_config': {'activation': model.activation_name},
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    meta = MANIFEST.get(Path(path).stem, {})
    resolved_activation = meta.get('activation') or checkpoint.get('model_config', {}).get('activation')
    if resolved_activation:
        model.set_activation(resolved_activation)
    # Older checkpoints stored the intermediate factorization maps.
    for key in ('base_idx_map', 'rn_src_idx', 'rn_dest_idx', 'st_src_idx', 'st_dest_idx'):
        state_dict.pop(key, None)
    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('iteration', 0)
