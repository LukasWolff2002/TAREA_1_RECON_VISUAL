"""
models/backbone.py

Backbone iDoc (ViT-Base) congelado para deteccion multi-escala.

Checkpoint real tiene este patron LoRA (confirmado por diagnostico):
    blocks.X.attn.q_proj.weight:   [768, 768]  <- peso base
    blocks.X.attn.q_proj.bias:     [768]
    blocks.X.attn.q_proj.w_lora_A: [64, 768]   <- down-proj (r=64)
    blocks.X.attn.q_proj.w_lora_B: [768, 64]   <- up-proj
    idem para k_proj y v_proj
    blocks.X.attn.out_proj.weight (o proj.weight): [768, 768]

Fusion LoRA: W_merged = W_base + w_lora_B @ w_lora_A
QKV resultante = cat([W_q, W_k, W_v], dim=0) -> [2304, 768]
"""

import sys, os, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

IDOC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "iDoc"))

def _load_idoc_module(filename, module_name):
    path = os.path.join(IDOC_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontro {filename} en {IDOC_DIR}")
    if IDOC_DIR not in sys.path:
        sys.path.insert(0, IDOC_DIR)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

_vit_module = _load_idoc_module("models/vision_transformer.py", "idoc_vision_transformer")
VisionTransformer = _vit_module.VisionTransformer


def _merge_lora_weights(state: dict, embed_dim: int = 768, depth: int = 12) -> dict:
    """
    Convierte el state_dict LoRA al formato estandar del ViT.

    Patron del checkpoint:
        blocks.X.attn.{q,k,v}_proj.weight   [768, 768]
        blocks.X.attn.{q,k,v}_proj.bias     [768]
        blocks.X.attn.{q,k,v}_proj.w_lora_A [r,   768]
        blocks.X.attn.{q,k,v}_proj.w_lora_B [768, r  ]
        blocks.X.attn.out_proj.weight        [768, 768]  (o proj.weight)
        blocks.X.attn.out_proj.bias          [768]       (o proj.bias)

    Resultado:
        blocks.X.attn.qkv.weight  [2304, 768]
        blocks.X.attn.qkv.bias    [2304]
        blocks.X.attn.proj.weight [768, 768]
        blocks.X.attn.proj.bias   [768]
    """
    new_state = {}

    for k, v in state.items():
        # Saltar keys LoRA y proj separadas (las procesamos abajo)
        if any(s in k for s in [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "w_lora_A", "w_lora_B"
        ]):
            continue
        new_state[k] = v

    n_merged = 0
    for i in range(depth):
        pref = f"blocks.{i}.attn"

        # Verificar que existan las keys necesarias
        if f"{pref}.q_proj.weight" not in state:
            continue

        merged_weights = []
        merged_biases  = []

        for proj in ["q_proj", "k_proj", "v_proj"]:
            W_base = state[f"{pref}.{proj}.weight"]          # [768, 768]
            bias   = state.get(f"{pref}.{proj}.bias",
                               torch.zeros(embed_dim))        # [768]

            # Fusion LoRA: W = W_base + w_lora_B @ w_lora_A
            if f"{pref}.{proj}.w_lora_A" in state:
                lora_A = state[f"{pref}.{proj}.w_lora_A"]   # [r, 768]
                lora_B = state[f"{pref}.{proj}.w_lora_B"]   # [768, r]
                W_merged = W_base + lora_B @ lora_A         # [768, 768]
            else:
                W_merged = W_base

            merged_weights.append(W_merged)
            merged_biases.append(bias)

        # QKV fusionado
        new_state[f"{pref}.qkv.weight"] = torch.cat(merged_weights, dim=0)  # [2304, 768]
        new_state[f"{pref}.qkv.bias"]   = torch.cat(merged_biases,  dim=0)  # [2304]
        n_merged += 1

        # Proyeccion de salida: out_proj o proj
        for out_name in ["out_proj", "proj"]:
            w_key = f"{pref}.{out_name}.weight"
            b_key = f"{pref}.{out_name}.bias"
            if w_key in state:
                new_state[f"{pref}.proj.weight"] = state[w_key]
                new_state[f"{pref}.proj.bias"]   = state.get(
                    b_key, torch.zeros(embed_dim))
                break

    if n_merged > 0:
        print(f"[iDocBackbone] LoRA fusionado en {n_merged} bloques (r=64).")
    return new_state


class iDocBackbone(nn.Module):
    """ViT-Base congelado con Simple Feature Pyramid (strides 4,8,16,32)."""

    def __init__(self, arch="vit_base", patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, extract_layers=None, pretrained_path=None):
        super().__init__()
        self.patch_size     = patch_size
        self.embed_dim      = embed_dim
        self.extract_layers = extract_layers or [2, 5, 8, 11]
        self.out_channels   = [embed_dim] * len(self.extract_layers)

        vit = VisionTransformer(
            img_size=[224], patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
            return_all_tokens=True, masked_im_modeling=False,
        )

        if pretrained_path and os.path.isfile(pretrained_path):
            ckpt  = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)

            # Detectar y fusionar LoRA
            has_lora = any("w_lora_A" in k or "q_proj" in k for k in state)
            if has_lora:
                print(f"[iDocBackbone] Checkpoint LoRA detectado — fusionando pesos...")
                state = _merge_lora_weights(state, embed_dim, depth)
            
            # Filtrar keys irrelevantes
            vit_state = {k: v for k, v in state.items()
                         if not k.startswith(("head.", "fc_norm", "mask_token"))}

            msg = vit.load_state_dict(vit_state, strict=False)
            important_missing = [
                k for k in msg.missing_keys
                if not any(s in k for s in ["head", "masked_embed", "fc_norm"])
            ]
            if important_missing:
                print(f"[iDocBackbone] Missing keys ({len(important_missing)}): "
                      f"{important_missing[:3]}{'...' if len(important_missing)>3 else ''}")
            else:
                print(f"[iDocBackbone] Todos los pesos cargados correctamente.")
        else:
            print("[iDocBackbone] ADVERTENCIA: pretrained_path no encontrado.")

        self.vit = vit
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.level_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in self.extract_layers
        ])

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def _extract_intermediate(self, x):
        tokens = self.vit.prepare_tokens(x)
        feats  = {}
        for i, blk in enumerate(self.vit.blocks):
            tokens = blk(tokens)
            if i in self.extract_layers:
                feats[i] = tokens[:, 1:, :]
        return [feats[i] for i in self.extract_layers]

    def forward(self, x):
        B, _, H, W = x.shape
        Hb, Wb = H // self.patch_size, W // self.patch_size
        raw = self._extract_intermediate(x)
        out = []
        for tokens, norm, stride in zip(raw, self.level_norms, [4, 8, 16, 32]):
            f = norm(tokens).transpose(1, 2).reshape(B, self.embed_dim, Hb, Wb)
            if   stride == 4:  f = F.interpolate(f, scale_factor=4.0, mode="bilinear", align_corners=False)
            elif stride == 8:  f = F.interpolate(f, scale_factor=2.0, mode="bilinear", align_corners=False)
            elif stride == 32: f = F.max_pool2d(f, kernel_size=2, stride=2)
            out.append(f)
        return out


class iDocQueryEncoder(nn.Module):
    def __init__(self, backbone, query_dim=768):
        super().__init__()
        self.backbone  = backbone
        self.query_dim = query_dim

    @torch.no_grad()
    def forward(self, query_img):
        vit    = self.backbone.vit
        tokens = vit.prepare_tokens(query_img)
        for blk in vit.blocks:
            tokens = blk(tokens)
        return vit.norm(tokens)[:, 0]