# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
import subprocess

import pytest
import torch
from torchcodec.decoders import Demuxer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_raw_frame_pict_type_in_display_order(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg and ffprobe")
    encoders = subprocess.check_output(["ffmpeg", "-v", "error", "-encoders"], text=True)
    if "libx264 " not in encoders:
        pytest.skip("Requires the libx264 encoder")

    path = tmp_path / "reordered.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=320x192:rate=30", "-frames:v", "90", "-c:v", "libx264",
            "-bf", "3", "-g", "90", "-x264-params", "b-adapt=0:scenecut=0",
            "-force_key_frames", "expr:eq(n,19)+eq(n,53)", str(path),
        ],
        check=True,
    )  # fmt: skip
    reference = json.loads(subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "frame=best_effort_timestamp_time,pict_type", "-of", "json", str(path),
        ]
    ))["frames"]
    assert {f["pict_type"] for f in reference} == {"I", "P", "B"}

    demuxer = Demuxer(path)
    decoder = demuxer.streams[0].make_decoder(device="cuda")
    decoded = [
        (frame.pts_seconds, frame.pict_type)
        for packet in demuxer
        for frame in decoder.decode(packet)
    ]
    tail = decoder.drain()
    assert tail
    decoded.extend((frame.pts_seconds, frame.pict_type) for frame in tail)
    assert len(decoded) == len(reference)
    for (pts, pict_type), expected in zip(decoded, reference):
        assert pict_type == expected["pict_type"]
        assert pts == pytest.approx(float(expected["best_effort_timestamp_time"]), abs=1e-6)
