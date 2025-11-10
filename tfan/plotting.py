"""Plotting helpers for TFAN experiments using a minimalist PNG backend."""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

Color = Tuple[int, int, int]


def _write_png(path: Path, width: int, height: int, pixels: Sequence[int]) -> None:
    """Write raw RGB pixels to a PNG file using the standard chunk format."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return length + chunk_type + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = []
    stride = width * 3
    for y in range(height):
        start = y * stride
        end = start + stride
        rows.append(b"\x00" + bytes(pixels[start:end]))
    compressed = zlib.compress(b"".join(rows), 9)

    png_bytes = bytearray()
    png_bytes.extend(signature)
    png_bytes.extend(_chunk(b"IHDR", header))
    png_bytes.extend(_chunk(b"IDAT", compressed))
    png_bytes.extend(_chunk(b"IEND", b""))

    path.write_bytes(png_bytes)


@dataclass
class PlotCanvas:
    width: int = 480
    height: int = 320
    margin: int = 32
    background: Color = (255, 255, 255)
    axis_color: Color = (0, 0, 0)

    def __post_init__(self) -> None:
        self.pixels: List[int] = list(self.background) * (self.width * self.height)

    def _index(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = self._index(x, y)
            self.pixels[idx : idx + 3] = list(color)

    def draw_axes(self) -> Tuple[int, int, int, int]:
        left = self.margin
        right = self.width - self.margin - 1
        top = self.margin
        bottom = self.height - self.margin - 1
        for x in range(left, right + 1):
            self.set_pixel(x, bottom, self.axis_color)
        for y in range(top, bottom + 1):
            self.set_pixel(left, y, self.axis_color)
        return left, top, right, bottom

    def draw_point(self, x: int, y: int, color: Color, size: int = 2) -> None:
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                self.set_pixel(x + dx, y + dy, color)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def save(self, path: Path) -> Path:
        _write_png(path, self.width, self.height, self.pixels)
        return path


def _scale(
    value: float,
    min_val: float,
    max_val: float,
    pixel_min: int,
    pixel_max: int,
    *,
    invert: bool = False,
) -> int:
    if math.isclose(max_val, min_val):
        ratio = 0.5
    else:
        ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    if invert:
        ratio = 1 - ratio
    pixel = pixel_min + ratio * (pixel_max - pixel_min)
    return int(round(pixel))


def save_pareto_front_plot(
    metrics: Mapping[str, float],
    dissipation: Iterable[float],
    output_dir: Path,
    *,
    suffix: str = "default",
) -> Path:
    values = list(metrics.values())
    labels = list(metrics.keys())
    dissipation = list(dissipation)
    if not values:
        values = [0.5]
        dissipation = [0.1]
        labels = ["metric"]

    canvas = PlotCanvas()
    left, top, right, bottom = canvas.draw_axes()
    min_x = min(dissipation)
    max_x = max(dissipation)
    min_y = min(values)
    max_y = max(values)

    for _label, x_val, y_val in zip(labels, dissipation, values):
        x = _scale(x_val, min_x, max_x, left, right)
        y = _scale(y_val, min_y, max_y, bottom, top, invert=True)
        canvas.draw_point(x, y, (52, 152, 219))
        # Draw a short tick to hint at the label position.
        canvas.draw_line(x, bottom, x, bottom + 4, (80, 80, 80))

    output_path = output_dir / f"pareto_acc_vs_diss_{suffix}.png"
    return canvas.save(output_path)


def save_throughput_plot(
    k_values: Iterable[int], throughput: Iterable[float], output_dir: Path, *, suffix: str = "default"
) -> Path:
    k_list = list(k_values)
    throughput_list = list(throughput)
    if not k_list:
        k_list = [1]
        throughput_list = [1.0]

    canvas = PlotCanvas()
    left, top, right, bottom = canvas.draw_axes()

    min_k = min(k_list)
    max_k = max(k_list)
    min_tp = min(throughput_list)
    max_tp = max(throughput_list)

    points = []
    for k, tp in zip(k_list, throughput_list):
        x = _scale(float(k), float(min_k), float(max_k), left, right)
        y = _scale(tp, min_tp, max_tp, bottom, top, invert=True)
        points.append((x, y))

    for idx in range(1, len(points)):
        canvas.draw_line(points[idx - 1][0], points[idx - 1][1], points[idx][0], points[idx][1], (39, 174, 96))
    for x, y in points:
        canvas.draw_point(x, y, (39, 174, 96))

    output_path = output_dir / f"throughput_vs_k_{suffix}.png"
    return canvas.save(output_path)


def save_nfl_curriculum_plot(
    structured: Iterable[float],
    scrambled: Iterable[float],
    output_dir: Path,
    *,
    suffix: str = "default",
) -> Path:
    structured_list = list(structured)
    scrambled_list = list(scrambled)
    steps = list(range(1, len(structured_list) + 1))
    if not steps:
        structured_list = [0.5]
        scrambled_list = [0.4]
        steps = [1]

    canvas = PlotCanvas()
    left, top, right, bottom = canvas.draw_axes()

    min_step = min(steps)
    max_step = max(steps)
    min_val = min(min(structured_list), min(scrambled_list))
    max_val = max(max(structured_list), max(scrambled_list))

    def _plot_line(values: Sequence[float], color: Color) -> None:
        pts: List[Tuple[int, int]] = []
        for step, val in zip(steps, values):
            x = _scale(float(step), float(min_step), float(max_step), left, right)
            y = _scale(val, min_val, max_val, bottom, top, invert=True)
            pts.append((x, y))
        for idx in range(1, len(pts)):
            canvas.draw_line(pts[idx - 1][0], pts[idx - 1][1], pts[idx][0], pts[idx][1], color)
        for x, y in pts:
            canvas.draw_point(x, y, color, size=1)

    _plot_line(structured_list, (231, 76, 60))
    _plot_line(scrambled_list, (44, 62, 80))

    output_path = output_dir / "nfl_structured_vs_scrambled.png"
    return canvas.save(output_path)
