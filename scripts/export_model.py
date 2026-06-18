#!/usr/bin/env python3
"""
Export the Kokoro-82M TorchScript model and voice packs for the C++ LibTorch backend.

What this script does:
  1. Downloads kokoro-v1_0.pth and voice .pt files from HuggingFace
  2. Wraps the model in KokoroWrapper (speed baked in at export time)
  3. Traces with torch.jit.trace and saves kokoro_traced.pt
  4. Converts each voice .pt tensor to a raw float32 .bin file
     (shape N×1×256, written as little-endian float32)

Requirements:
    pip install "transformers<5.0.0" spacy torch huggingface_hub
    pip install kokoro  # or clone hexgrad/Kokoro-82M

Usage:
    python scripts/export_model.py [--output-dir models] [--speed 1.0]
"""

from istftnet import Decoder
from modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

@dataclass
class PhonemeTimestamp:
    token_id: str
    start: float
    end: float
    duration_frames: int

@dataclass
class Output:
    audio: torch.FloatTensor
    pred_dur: Optional[torch.LongTensor] = None
    phoneme_timestamps: Optional[list] = None

class KModel(torch.nn.Module):
    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple:
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

        return audio, pred_dur


class KokoroWrapper(torch.nn.Module):
    """
    Wraps KModel for TorchScript tracing.

    Returns (audio, ts_tensor) where ts_tensor has shape [seq_len, 4]:
        col 0: token_id  (float, cast from input_ids)
        col 1: start_sec
        col 2: end_sec
        col 3: duration_frames

    IMPORTANT: ts_tensor is built entirely from tensors (no Python lists or
    dataclass comprehensions) so torch.jit.trace generalises correctly to
    sequences of any length at runtime.
    """

    HOP_LENGTH  = 300
    SAMPLE_RATE = 24000

    def __init__(self, m: KModel, speed: float = 1.0):
        super().__init__()
        self.m     = m
        self.speed = speed

    def forward(
        self,
        input_ids: torch.Tensor,
        ref_s:     torch.Tensor,
    ) -> tuple:  # (audio: Tensor, ts_tensor: Tensor[seq_len, 4])

        audio, pred_dur = self.m.forward_with_tokens(input_ids, ref_s, self.speed)

        # --- Build timestamp tensor entirely in tensor ops ---
        # pred_dur shape: [seq_len]  (one duration-in-frames per input token)
        seq_len = pred_dur.shape[0]

        # seconds per frame
        spf = self.HOP_LENGTH / self.SAMPLE_RATE  # scalar, constant

        # duration in seconds for each token
        dur_sec = pred_dur.float() * spf          # [seq_len]

        # cumulative end times
        end_sec   = torch.cumsum(dur_sec, dim=0)  # [seq_len]
        start_sec = end_sec - dur_sec             # [seq_len]

        # token ids as float (input_ids shape is [1, seq_len])
        token_ids_f = input_ids[0].float()        # [seq_len]

        # Stack into [seq_len, 4]: [token_id, start, end, dur_frames]
        ts_tensor = torch.stack(
            [token_ids_f, start_sec, end_sec, pred_dur.float()],
            dim=1
        )  # [seq_len, 4]

        return audio, ts_tensor


import argparse
import os
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description="Export Kokoro-82M to TorchScript")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save the exported model and voices")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier baked into the trace (default 1.0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    voices_dir = os.path.join(args.output_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)

    try:
        import torch
    except ImportError:
        print("Missing torch. Install with: pip install torch")
        sys.exit(1)

    try:
        import spacy
    except ImportError:
        import types
        spacy_stub = types.ModuleType("spacy")
        spacy_stub.__version__ = "3.0.0"
        sys.modules["spacy"] = spacy_stub

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Missing huggingface_hub. Install with: pip install huggingface_hub")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Device: {device}  Speed: {args.speed}")

    # ── Download weights ─────────────────────────────────────────────────────
    print("Downloading kokoro-v1_0.pth from HuggingFace...")
    try:
        weights_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print(f"Loading weights from {weights_path}")
    model = KModel().to(device).eval()

    # ── Wrap and trace ───────────────────────────────────────────────────────
    wrapper = KokoroWrapper(model, speed=args.speed).to(device).eval()

    seq_len     = 50
    example_ids = torch.randint(1, 160, (1, seq_len), dtype=torch.long).to(device)
    example_ref = torch.zeros(1, 256, dtype=torch.float32).to(device)

    out_path = os.path.join(args.output_dir, "kokoro_traced.pt")
    print(f"Tracing model (seq_len={seq_len}, speed={args.speed})...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_ids, example_ref), strict=False)
    traced.save(out_path)
    print(f"  Saved: {out_path}")

    # ── Download and convert voice packs ────────────────────────────────────
    print("\nDownloading voice packs...")
    try:
        voice_repo = snapshot_download("hexgrad/Kokoro-82M",
                                        allow_patterns=["voices/*.pt"])
    except Exception as e:
        print(f"  Warning: snapshot_download failed ({e})")
        voice_repo = None

    import glob

    if voice_repo:
        pt_files = glob.glob(os.path.join(voice_repo, "voices", "*.pt"))
    else:
        pt_files = []

    if not pt_files:
        pt_files = glob.glob(os.path.join(voices_dir, "*.pt"))
        if pt_files:
            print("  Found .pt files already in voices dir, converting...")

    if not pt_files:
        print("  No voice .pt files found. Copy them manually to", voices_dir)
        print("  Then re-run this script to convert them to .bin format.")
    else:
        _convert_voices(pt_files, voices_dir, device)

    print(f"""
Export complete!
  Model:  {out_path}
  Voices: {voices_dir}/*.bin

Build:
  mkdir build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
  make -j$(nproc)

Run:
  LD_LIBRARY_PATH=/path/to/libtorch/lib ./kokoro_tts \\
      --model {out_path} \\
      --voices {voices_dir} \\
      --dict data/cmudict.dict \\
      "Hello, world!"
""")


def _convert_voices(pt_files, voices_dir, device):
    import torch

    for pt_path in sorted(pt_files):
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        bin_path = os.path.join(voices_dir, stem + ".bin")

        try:
            tensor = torch.load(pt_path, map_location=device, weights_only=True)
        except Exception:
            try:
                tensor = torch.load(pt_path, map_location=device, weights_only=False)
            except Exception as e:
                print(f"  Skipping {stem}: {e}")
                continue

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)

        if tensor.dim() != 3 or tensor.shape[1] != 1 or tensor.shape[2] != 256:
            print(f"  Skipping {stem}: unexpected shape {tuple(tensor.shape)}")
            continue

        floats = tensor.to(torch.float32).cpu().contiguous().numpy()
        with open(bin_path, "wb") as f:
            f.write(floats.tobytes())

        n = tensor.shape[0]
        print(f"  {stem}: shape {tuple(tensor.shape)} → {bin_path}  ({n*256*4} bytes)")


if __name__ == "__main__":
    main()