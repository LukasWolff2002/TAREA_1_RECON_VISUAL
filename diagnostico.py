"""
diagnostico.py
Ejecutar desde la raíz del proyecto:
    python diagnostico.py --pretrained idoc_pretrained.pth --dataset detection_dataset_sketches.json

Revisa:
  1. Estructura del checkpoint (qué keys tiene)
  2. Que el split del dataset funcione
  3. Que los imports de train_fcos sean los correctos
"""

import sys, os, json, argparse, random
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="idoc_pretrained.pth")
    p.add_argument("--dataset",    default="detection_dataset_sketches.json")
    return p.parse_args()


def check_checkpoint(path):
    print("\n" + "="*60)
    print("CHECKPOINT:", path)
    print("="*60)
    if not os.path.isfile(path):
        print("  ERROR: archivo no encontrado"); return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # Determinar dónde están los pesos
    if isinstance(ckpt, dict):
        for key in ["state_dict", "teacher", "student", "model"]:
            if key in ckpt:
                weights = ckpt[key]
                all_keys = list(weights.keys())
                print(f"\n  Pesos bajo '{key}': {len(all_keys)} keys")
                print(f"  Primeras 10 keys:")
                for k in all_keys[:10]: print(f"    {k}: {weights[k].shape}")

                # Buscar qkv, attn, backbone
                qkv_keys   = [k for k in all_keys if "qkv" in k]
                attn_keys  = [k for k in all_keys if "attn" in k]
                bb_keys    = [k for k in all_keys if k.startswith("backbone")]

                print(f"\n  Keys con 'backbone': {len(bb_keys)}")
                print(f"  Keys con 'attn':     {len(attn_keys)}")
                print(f"  Keys con 'qkv':      {len(qkv_keys)}")
                if qkv_keys:
                    print(f"  Ejemplo qkv key: {qkv_keys[0]}: {weights[qkv_keys[0]].shape}")

                # Detectar si es LoRA
                lora_keys = [k for k in all_keys if any(s in k for s in
                             ["lora_A", "lora_B", "qkv_A", "qkv_B", "W_A", "W_B"])]
                print(f"  Keys LoRA:           {len(lora_keys)}")
                if lora_keys:
                    print("  *** MODELO LORA DETECTADO ***")
                    for k in lora_keys[:5]:
                        print(f"    {k}: {weights[k].shape}")

                # Detectar patch_embed para confirmar arch
                pe_keys = [k for k in all_keys if "patch_embed" in k]
                for k in pe_keys[:3]:
                    print(f"  {k}: {weights[k].shape}")
                break
    else:
        print(f"  El checkpoint no es un dict: {type(ckpt)}")


def check_dataset_split(path):
    print("\n" + "="*60)
    print("DATASET SPLIT:", path)
    print("="*60)
    if not os.path.isfile(path):
        print("  ERROR: archivo no encontrado"); return

    with open(path) as f:
        data = json.load(f)

    print(f"  Top-level keys: {list(data.keys()) if isinstance(data, dict) else 'lista'}")
    samples = data.get("samples", data if isinstance(data, list) else [])
    print(f"  Total samples: {len(samples)}")

    page_ids = list({s["page_id"] for s in samples})
    print(f"  Page_ids únicos: {len(page_ids)}")

    rng = random.Random(42)
    rng.shuffle(page_ids)
    n = len(page_ids)
    n_train = int(n * 0.8); n_val = int(n * 0.1)
    train_pages = set(page_ids[:n_train])
    val_pages   = set(page_ids[n_train:n_train+n_val])
    test_pages  = set(page_ids[n_train+n_val:])
    train_ids = [s["sample_id"] for s in samples if s["page_id"] in train_pages]
    val_ids   = [s["sample_id"] for s in samples if s["page_id"] in val_pages]
    test_ids  = [s["sample_id"] for s in samples if s["page_id"] in test_pages]
    print(f"  Split esperado → train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")


def check_imports():
    print("\n" + "="*60)
    print("IMPORTS DE train_fcos")
    print("="*60)
    try:
        from train_fcos.datasets import build_datasets
        import inspect
        src_file = inspect.getfile(build_datasets)
        print(f"  build_datasets importado desde: {src_file}")
        src = inspect.getsource(build_datasets)
        has_split_print = "Split:" in src
        has_filter      = "sample_ids" in src
        print(f"  Tiene print 'Split:':      {has_split_print}")
        print(f"  Tiene filtro sample_ids:   {has_filter}")
        if not has_split_print or not has_filter:
            print("  *** ARCHIVO DESACTUALIZADO — reemplazar con versión nueva ***")
    except Exception as e:
        print(f"  ERROR al importar: {e}")

    try:
        from train_fcos.models import FCOSDetector
        import inspect
        src_file = inspect.getfile(FCOSDetector)
        print(f"  FCOSDetector importado desde: {src_file}")
    except Exception as e:
        print(f"  ERROR al importar FCOSDetector: {e}")


if __name__ == "__main__":
    args = parse_args()
    check_checkpoint(args.pretrained)
    check_dataset_split(args.dataset)
    check_imports()
    print("\nDiagnóstico completo.")