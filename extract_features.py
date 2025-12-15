#!/usr/bin/env python3
"""
extract_features.py

Feature extraction pipeline for FPGA bitstreams (tilegrid-lite friendly).
"""

import argparse
import os
import struct
import glob
import json
from pathlib import Path
from collections import Counter, defaultdict
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- Configuration ----------------
FRAME_WORDS = 101
WORD_BYTES = 4
FRAME_BYTES = FRAME_WORDS * WORD_BYTES
SYNC_WORD = b'\xAA\x99\x55\x66'

# ---------------- Utilities ----------------
def read_file_bytes(path):
    with open(path, 'rb') as f:
        return f.read()

def find_sync_offset(data):
    idx = data.find(SYNC_WORD)
    if idx == -1:
        return None
    return idx + len(SYNC_WORD)

def chunk_frames_from_offset(data, offset):
    frames = []
    payload = data[offset:]
    for i in range(0, len(payload), FRAME_BYTES):
        frames.append(payload[i:i+FRAME_BYTES])
    return frames

def approximate_frames(data):
    off = find_sync_offset(data)
    if off is not None:
        frames = chunk_frames_from_offset(data, off)
        if len(frames) > 0:
            return frames
    return chunk_frames_from_offset(data, 128)

def bits_in_bytes(bs):
    arr = np.frombuffer(bs, dtype=np.uint8)
    return int(np.unpackbits(arr).sum())

def bit_density(bs):
    total_bits = len(bs) * 8
    if total_bits == 0:
        return 0.0
    return bits_in_bytes(bs) / total_bits

def shannon_entropy(bs):
    if len(bs) == 0:
        return 0.0
    arr = np.frombuffer(bs, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def words32_from_bytes(bs, endian='big'):
    if len(bs) % 4 != 0:
        bs = bs + b'\x00' * (4 - (len(bs) % 4))
    words = []
    for i in range(0, len(bs), 4):
        words.append(int.from_bytes(bs[i:i+4], endian))
    return words

def hamming_bytes(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    wa = words32_from_bytes(a[:n])
    wb = words32_from_bytes(b[:n])
    dist = sum(bin(x ^ y).count("1") for x, y in zip(wa, wb))
    if len(a) != len(b):
        dist += bits_in_bytes(a[n:] if len(a) > len(b) else b[n:])
    return dist

# ---------------- Tilegrid ----------------
def load_tilegrid_mapping(path):
    with open(path, 'r') as f:
        data = json.load(f)
    mapping = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                mapping[int(k)] = v
            except:
                pass
    elif isinstance(data, list):
        for e in data:
            for i in range(e['start_frame'], e['end_frame'] + 1):
                mapping[i] = e['tile_type']
    return mapping

def aggregate_by_tile_type(frames, mapping):
    agg = defaultdict(list)
    for i, f in enumerate(frames):
        agg[mapping.get(i, "UNKNOWN")].append(f)
    return agg

# ---------------- Feature Extraction ----------------
def extract_features_for_bitstream(path, baseline_frames=None, tilegrid_map=None, save_frame_csv_dir=None):
    name = Path(path).name
    data = read_file_bytes(path)
    frames = approximate_frames(data)
    num_frames = len(frames)

    per_frame_density = [bit_density(f) for f in frames]
    per_frame_ones = [bits_in_bytes(f) for f in frames]
    per_frame_entropy = [shannon_entropy(f) for f in frames]

    words = words32_from_bytes(b''.join(frames))
    word_counts = Counter(words)
    repeated_word_count = sum(1 for v in word_counts.values() if v > 1)

    if baseline_frames is not None:
        m = min(len(frames), len(baseline_frames))
        hd_list = [hamming_bytes(frames[i], baseline_frames[i]) for i in range(m)]
    else:
        hd_list = [hamming_bytes(frames[i], frames[i-1]) for i in range(1, len(frames))]

    inter_frame_hd = [hamming_bytes(frames[i], frames[i-1]) for i in range(1, len(frames))]

    active_frames = [d for d in per_frame_density if d > 0.01]
    density_diffs = np.abs(np.diff(per_frame_density))

    feat = {}
    feat['file'] = name
    feat['num_frames'] = num_frames
    feat['mean_bit_density'] = float(np.mean(per_frame_density)) if per_frame_density else 0.0
    feat['var_bit_density'] = float(np.var(per_frame_density)) if per_frame_density else 0.0
    feat['mean_frame_ones'] = float(np.mean(per_frame_ones)) if per_frame_ones else 0.0
    feat['median_frame_ones'] = float(np.median(per_frame_ones)) if per_frame_ones else 0.0
    feat['repeated_word_count'] = int(repeated_word_count)

    feat['mean_frame_entropy'] = float(np.mean(per_frame_entropy)) if per_frame_entropy else 0.0
    feat['var_frame_entropy'] = float(np.var(per_frame_entropy)) if per_frame_entropy else 0.0
    feat['max_frame_entropy'] = float(np.max(per_frame_entropy)) if per_frame_entropy else 0.0

    feat['active_frame_ratio'] = len(active_frames) / num_frames if num_frames else 0.0
    feat['mean_density_jump'] = float(np.mean(density_diffs)) if len(density_diffs) else 0.0
    feat['max_density_jump'] = float(np.max(density_diffs)) if len(density_diffs) else 0.0

    if hd_list:
        feat['mean_hamming'] = float(np.mean(hd_list))
        feat['max_hamming'] = int(np.max(hd_list))
        feat['median_hamming'] = float(np.median(hd_list))
    else:
        feat['mean_hamming'] = 0.0
        feat['max_hamming'] = 0
        feat['median_hamming'] = 0.0

    feat['mean_inter_frame_hamming'] = float(np.mean(inter_frame_hd)) if inter_frame_hd else 0.0
    feat['max_inter_frame_hamming'] = int(np.max(inter_frame_hd)) if inter_frame_hd else 0

    if tilegrid_map:
        agg = aggregate_by_tile_type(frames, tilegrid_map)
        for t, fl in agg.items():
            key = f"tile_{t}"
            feat[f"{key}_frame_count"] = len(fl)
            dens = [bit_density(x) for x in fl]
            feat[f"{key}_mean_density"] = float(np.mean(dens)) if dens else 0.0
            feat[f"{key}_var_density"] = float(np.var(dens)) if dens else 0.0

    if save_frame_csv_dir:
        Path(save_frame_csv_dir).mkdir(parents=True, exist_ok=True)
        rows = []
        for i, f in enumerate(frames):
            rows.append({
                'frame_index': i,
                'bit_density': bit_density(f),
                'ones_count': bits_in_bytes(f),
                'entropy': shannon_entropy(f)
            })
        pd.DataFrame(rows).to_csv(Path(save_frame_csv_dir) / f"{name}_frames.csv", index=False)

    return feat, frames

# ---------------- Orchestration ----------------
def gather_bitstreams(data_dir):
    files = []
    for label in ['benign', 'malicious']:
        for p in Path(data_dir, label).glob("*.bit"):
            files.append((str(p), label))
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--out', default='features.csv')
    parser.add_argument('--tilegrid', default=None)
    parser.add_argument('--frame-csv-dir', default=None)
    parser.add_argument('--baseline', default=None)
    args = parser.parse_args()

    files = gather_bitstreams(args.data_dir)
    tilegrid_map = load_tilegrid_mapping(args.tilegrid) if args.tilegrid else None

    baseline_frames = None
    if args.baseline and os.path.exists(args.baseline):
        baseline_frames = approximate_frames(read_file_bytes(args.baseline))
    else:
        for f, l in files:
            if l == 'benign':
                baseline_frames = approximate_frames(read_file_bytes(f))
                break

    feats = []
    for path, label in tqdm(files):
        feat, _ = extract_features_for_bitstream(
            path,
            baseline_frames=baseline_frames,
            tilegrid_map=tilegrid_map,
            save_frame_csv_dir=args.frame_csv_dir
        )
        feat['label'] = label
        feat['path'] = path
        feat['filesize_bytes'] = os.path.getsize(path)
        feats.append(feat)

    df = pd.DataFrame(feats)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
