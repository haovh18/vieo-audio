#!/usr/bin/env python3
"""Offline single-sample forward validation for local MuQ / MuQ-MuLan."""

import argparse
import csv
import json
import os
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torchaudio
from safetensors import safe_open
from muq import MuQ, MuQMuLan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "outputs/muq_videomae/sample_index.csv"
DEFAULT_OUTPUT = ROOT / "outputs/muq_videomae/checks/muq_single_report.json"
MUQ_DIR = ROOT / "weights/muq/MuQ-large-msd-iter"
MULAN_DIR = ROOT / "weights/muq/MuQ-MuLan-large"
MODEL_SR = 24000
MIN_SECONDS = 10


def tensor_info(tensor):
    value = tensor.detach()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "all_finite": bool(torch.isfinite(value).all().item()),
    }


def resolve_path(raw):
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def find_audio_field(row):
    for field in ("audio_path", "wav_path", "mp3_path"):
        value = row.get(field)
        if value:
            return field, value
    return None, None


def key_check(model, checkpoint):
    model_state = model.state_dict()
    model_keys = set(model_state)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
        mismatched = []
        for key in sorted(model_keys & checkpoint_keys):
            if tuple(model_state[key].shape) != tuple(handle.get_slice(key).get_shape()):
                mismatched.append(
                    {
                        "key": key,
                        "model_shape": list(model_state[key].shape),
                        "checkpoint_shape": list(handle.get_slice(key).get_shape()),
                    }
                )
    return {
        "missing_keys": sorted(model_keys - checkpoint_keys),
        "unexpected_keys": sorted(checkpoint_keys - model_keys),
        "mismatched_keys": mismatched,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "started",
        "strictly_offline": True,
        "training": False,
        "index_path": str(index_path.relative_to(ROOT)),
    }
    try:
        with index_path.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        audio_field, audio_value = find_audio_field(row)
        report["sample_id"] = row.get("sample_id", row.get("id"))
        report["audio_field"] = audio_field
        if audio_field is None:
            report.update(
                status="failed_no_audio_field",
                error="No audio_path, wav_path, or mp3_path field with a value.",
                available_fields=sorted(row),
            )
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit(1)

        audio_path = resolve_path(audio_value)
        report["audio_path"] = str(audio_path.relative_to(ROOT))
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        info = torchaudio.info(str(audio_path))
        waveform, original_sr = torchaudio.load(str(audio_path))
        report["audio"] = {
            "original_sample_rate": original_sr,
            "original_channels": waveform.shape[0],
            "original_num_frames": waveform.shape[1],
            "duration_seconds": waveform.shape[1] / original_sr,
            "encoding": info.encoding,
        }
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != MODEL_SR:
            waveform = torchaudio.functional.resample(waveform, original_sr, MODEL_SR)
        required_samples = MODEL_SR * MIN_SECONDS
        length_action = "none"
        if waveform.shape[1] < required_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, required_samples - waveform.shape[1])
            )
            length_action = "zero_pad_to_10_seconds"
        elif waveform.shape[1] > required_samples:
            waveform = waveform[:, :required_samples]
            length_action = "truncate_to_first_10_seconds"
        report["preprocessing"] = {
            "model_sample_rate": MODEL_SR,
            "mono": True,
            "minimum_seconds": MIN_SECONDS,
            "length_action": length_action,
            "input_tensor": tensor_info(waveform),
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        report["device"] = str(device)

        try:
            muq = MuQ.from_pretrained(str(MUQ_DIR))
            muq_loading = key_check(muq, MUQ_DIR / "model.safetensors")
            muq = muq.to(device).eval()
            model_input = waveform.to(device)
            with torch.no_grad():
                output = muq(model_input, output_hidden_states=True)
                final_embedding = output.last_hidden_state.mean(dim=1)
            hidden_states = output.hidden_states
            report["muq"] = {
                "status": "pass",
                "checkpoint": "weights/muq/MuQ-large-msd-iter/model.safetensors",
                "checkpoint_loaded": True,
                "loading_info": muq_loading,
                "input_tensor": tensor_info(model_input),
                "last_hidden_state": tensor_info(output.last_hidden_state),
                "final_embedding_pooling": "mean over time of last_hidden_state",
                "final_embedding": tensor_info(final_embedding),
                "hidden_states_available": hidden_states is not None,
                "hidden_states_count": len(hidden_states),
                "hidden_state_shapes": [list(state.shape) for state in hidden_states],
                "hidden_states_all_finite": all(
                    torch.isfinite(state).all().item() for state in hidden_states
                ),
            }
        except Exception:
            report["muq"] = {
                "status": "fail",
                "checkpoint_loaded": False,
                "error": traceback.format_exc(),
            }
        finally:
            for name in ("muq", "output", "final_embedding", "model_input"):
                if name in locals():
                    del locals()[name]
            if device.type == "cuda":
                torch.cuda.empty_cache()

        try:
            mulan = MuQMuLan.from_pretrained(str(MULAN_DIR))
            mulan = mulan.to(device).eval()
            model_input = waveform.to(device)
            with torch.no_grad():
                mulan_embedding = mulan(wavs=model_input)
            report["muq_mulan"] = {
                "status": "pass",
                "checkpoint": "weights/muq/MuQ-MuLan-large/pytorch_model.bin",
                "checkpoint_loaded": True,
                "input_tensor": tensor_info(model_input),
                "final_embedding": tensor_info(mulan_embedding),
                "hidden_states_available": False,
                "hidden_states_count": 0,
                "hidden_state_shapes": [],
                "hidden_states_reason": (
                    "MuQMuLan.forward returns only the projected contrastive audio latent; "
                    "its public forward API does not return backbone hidden states."
                ),
                "loading_info": {
                    "missing_keys": None,
                    "unexpected_keys": None,
                    "mismatched_keys": None,
                    "reason": "PyTorchModelHubMixin does not expose loading_info.",
                },
            }
        except Exception:
            report["muq_mulan"] = {
                "status": "fail",
                "checkpoint": "weights/muq/MuQ-MuLan-large/pytorch_model.bin",
                "checkpoint_loaded": False,
                "final_embedding": None,
                "hidden_states_available": False,
                "hidden_states_count": 0,
                "hidden_state_shapes": [],
                "loading_info": {
                    "missing_keys": None,
                    "unexpected_keys": None,
                    "mismatched_keys": None,
                    "reason": "Model construction failed before checkpoint loading completed.",
                },
                "error": traceback.format_exc(),
            }

        muq_ok = report["muq"]["status"] == "pass"
        mulan_ok = report["muq_mulan"]["status"] == "pass"
        report["status"] = (
            "pass" if muq_ok and mulan_ok else
            "partial_pass_muq_only" if muq_ok else
            "fail"
        )
    except SystemExit:
        raise
    except Exception:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
    finally:
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
