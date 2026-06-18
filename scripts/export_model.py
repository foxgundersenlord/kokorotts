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
    token_id: str           # The actual phoneme (e.g., "K", "AA", "T")
    start: float        # Start time in seconds
    end: float          # End time in seconds
    duration_frames: int     # Length in seconds

@dataclass
class Output:
    audio: torch.FloatTensor
    pred_dur: Optional[torch.LongTensor] = None
    phoneme_timestamps: Optional[list[PhonemeTimestamp]] = None

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

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
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
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
        # --- ADD THIS INSIDE forward_with_tokens ---

    # 1. Configuration (Check your model's config for these)
        sample_rate = 24000
        hop_length = 300
        seconds_per_frame = hop_length / sample_rate

    # 2. Convert tensor to a list of integers
        durations_list = pred_dur.tolist()

    # 3. Calculate timestamps
        phoneme_timestamps = []
        current_time = 0.0

        for i, dur_in_frames in enumerate(durations_list):
            duration_secs = dur_in_frames * seconds_per_frame
            # Extract ID from the input_ids tensor
            t_id = input_ids[0, i].item()

            phoneme_timestamps.append(PhonemeTimestamp(
                token_id=t_id, # Ensure this is an int/float, not a string
                start=round(current_time, 4),
                end=round(current_time + duration_secs, 4),
                duration_frames=dur_in_frames
            ))
            current_time += duration_secs

        # Now you can return this alongside the audio
        return audio, pred_dur, phoneme_timestamps

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur, ts = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur, phoneme_timestamps=ts) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration



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

    # Stub out spacy if not installed — transformers tries to import it
    try:
        import spacy
    except ImportError:
        import types
        spacy_stub = types.ModuleType("spacy")
        spacy_stub.__version__ = "3.0.0"
        sys.modules["spacy"] = spacy_stub

#    try:
#        from kokoro import KModel
#    except ImportError:
#        print("Missing kokoro package.")
#        print("Clone hexgrad/Kokoro-82M and run: pip install -e .")
#        sys.exit(1)

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
    # weights_only=False required — checkpoint uses pickle, not safetensors
    #state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    #model.load_state_dict(state_dict)

    # ── Wrap model for tracing ───────────────────────────────────────────────
    # KokoroWrapper bakes speed into the forward pass so the C++ side only needs
    # to pass (input_ids, ref_s).  The model returns audio directly (no tuple).
    speed = args.speed

    class KokoroWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        # Explicitly type hint the return as a Tuple of two Tensors
        def forward(self, input_ids: torch.Tensor, ref_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Call the internal method
            audio, _, ts_list = self.m.forward_with_tokens(input_ids, ref_s, speed)

            # Convert phoneme list to a Tensor (C++ can't read Python lists/objects)
            # We pack [id, start, end, frames] into rows
            ts_tensor = torch.tensor([
                [float(p.token_id), p.start, p.end, float(p.duration_frames)]
                for p in ts_list
            ], dtype=torch.float32)

            # RETURN A TUPLE
            return audio, ts_tensor

    wrapper = KokoroWrapper(model).to(device).eval()

    # Use a realistic sequence length for tracing (not too short)
    seq_len = 50
    example_ids = torch.randint(1, 160, (1, seq_len), dtype=torch.long).to(device)
    example_ref = torch.zeros(1, 256, dtype=torch.float32).to(device)

    out_path = os.path.join(args.output_dir, "kokoro_traced.pt")
    print(f"Tracing model (seq_len={seq_len}, speed={speed})...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_ids, example_ref), strict=False)
    traced.save(out_path)
    print(f"  Saved: {out_path}")

    # ── Download voice packs ─────────────────────────────────────────────────
    print("\nDownloading voice packs...")
    try:
        voice_repo = snapshot_download("hexgrad/Kokoro-82M",
                                        allow_patterns=["voices/*.pt"])
    except Exception as e:
        print(f"  Warning: snapshot_download failed ({e})")
        print("  Trying individual file download as fallback...")
        voice_repo = None

    import glob, shutil

    if voice_repo:
        pt_files = glob.glob(os.path.join(voice_repo, "voices", "*.pt"))
    else:
        pt_files = []

    if not pt_files:
        # Last resort: check if files are already in output dir as .pt
        pt_files = glob.glob(os.path.join(voices_dir, "*.pt"))
        if pt_files:
            print("  Found .pt files already in voices dir, converting...")

    if not pt_files:
        print("  No voice .pt files found. Copy them manually to", voices_dir)
        print("  Then re-run this script to convert them to .bin format.")
    else:
        _convert_voices(pt_files, voices_dir, device)

    # ── Summary ───────────────────────────────────────────────────────────────
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
    """Convert .pt voice tensors to raw float32 .bin files.

    Voice tensors have shape (N, 1, 256) where N is typically 510.
    The C++ side reads them as a flat array of float32 values and
    reconstructs the shape as (total_floats // 256, 1, 256).
    """
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
            # Some voices are stored as (N, 256) — add the middle dim
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
