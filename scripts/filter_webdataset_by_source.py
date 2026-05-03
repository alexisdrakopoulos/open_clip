#!/usr/bin/env python3

"""Filter WebDataset tar shards by the value of their `.source` member.

The script scans input shards in parallel, writes temporary filtered shards per
input shard, then repacks all retained samples into a fresh dense shard series
with a configurable maximum number of samples per output shard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
import tarfile
import tempfile
from typing import Iterable, Iterator


LOGGER = logging.getLogger("filter_webdataset_by_source")
DEFAULT_MAX_SAMPLES_PER_SHARD = 1000
SampleMembers = list[tuple[tarfile.TarInfo, bytes]]


@dataclass(frozen=True)
class FilteredShardResult:
    input_path: str
    temp_path: str | None
    total_samples: int
    matched_samples: int
    written_bytes: int


class DenseShardWriter:
    def __init__(self, output_dir: Path, output_prefix: str, max_samples_per_shard: int):
        if max_samples_per_shard <= 0:
            raise ValueError("max_samples_per_shard must be > 0")

        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.max_samples_per_shard = max_samples_per_shard
        self.current_tar: tarfile.TarFile | None = None
        self.current_path: Path | None = None
        self.current_sample_count = 0
        self.next_shard_index = 0
        self.total_samples = 0
        self.total_written_bytes = 0
        self.shard_paths: list[str] = []
        self.shard_sample_counts: dict[str, int] = {}

    def _open_next_shard(self) -> None:
        self.close()
        self.current_path = self.output_dir / f"{self.output_prefix}_{self.next_shard_index:06d}.tar"
        self.current_tar = tarfile.open(self.current_path, "w")
        self.current_sample_count = 0
        self.shard_paths.append(str(self.current_path))
        self.shard_sample_counts[self.current_path.name] = 0
        self.next_shard_index += 1

    def add_sample(self, members: SampleMembers) -> None:
        if not members:
            return

        if self.current_tar is None or self.current_sample_count >= self.max_samples_per_shard:
            self._open_next_shard()

        sample_key = f"sample_{self.total_samples:09d}"
        for member, data in members:
            _, suffix = split_key_and_suffix(member.name)
            if suffix is None:
                continue
            member_name = f"{sample_key}.{suffix}"
            self.total_written_bytes += _copy_member(member, data, self.current_tar, name=member_name)

        self.current_sample_count += 1
        self.total_samples += 1
        assert self.current_path is not None
        self.shard_sample_counts[self.current_path.name] += 1

    def close(self) -> None:
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None
            self.current_path = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered WebDataset shard set containing only samples from a target source.",
    )
    parser.add_argument(
        "--input-shards",
        nargs="+",
        required=True,
        help=(
            "Input shard expression(s). Supports brace expansion, glob patterns, or multiple datasets "
            "joined with '::'. Example: '../../clip_webdataset/shard_{000000..003456}.tar'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where filtered shards will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default="shard",
        help="Prefix for the new dense output shards. Default: shard",
    )
    parser.add_argument(
        "--source",
        default="MET",
        help="Exact `.source` value to retain. Default: MET",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use. Default: all CPU cores.",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_SHARD,
        help="Maximum retained samples per output shard. Default: 1000",
    )
    parser.add_argument(
        "--manifest-name",
        default="shards.txt",
        help="Filename for the manifest of output shard paths inside the output directory.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=32,
        help="Log progress every N completed shards. Default: 32",
    )
    return parser.parse_args()


def _split_data_sources(values: Iterable[str]) -> list[str]:
    parts = []
    for value in values:
        parts.extend(part for part in value.split("::") if part)
    return parts


def _expand_brace_body(body: str) -> list[str]:
    range_parts = body.split("..")
    if len(range_parts) in {2, 3} and all(part for part in range_parts):
        start, end = range_parts[0], range_parts[1]
        step = int(range_parts[2]) if len(range_parts) == 3 else 1
        if step == 0:
            raise ValueError("Brace expansion step cannot be 0")

        if re.fullmatch(r"-?\d+", start) and re.fullmatch(r"-?\d+", end):
            start_int = int(start)
            end_int = int(end)
            width = max(len(start.lstrip("-")), len(end.lstrip("-")))
            if start_int > end_int and step > 0:
                step = -step
            values = []
            stop = end_int + (1 if step > 0 else -1)
            for value in range(start_int, stop, step):
                sign = "-" if value < 0 else ""
                values.append(f"{sign}{abs(value):0{width}d}")
            return values

        if len(start) == 1 and len(end) == 1:
            start_ord = ord(start)
            end_ord = ord(end)
            if start_ord > end_ord and step > 0:
                step = -step
            stop = end_ord + (1 if step > 0 else -1)
            return [chr(value) for value in range(start_ord, stop, step)]

    return [part for part in body.split(",") if part]


def brace_expand(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]

    prefix = pattern[:match.start()]
    suffix = pattern[match.end():]
    expanded_patterns = []
    for option in _expand_brace_body(match.group(1)):
        expanded_patterns.extend(brace_expand(f"{prefix}{option}{suffix}"))
    return expanded_patterns


def _expand_one_pattern(pattern: str) -> list[str]:
    paths = []
    for expanded in brace_expand(pattern):
        matches = sorted(glob.glob(expanded))
        if matches:
            paths.extend(matches)
        else:
            paths.append(expanded)
    return paths


def expand_input_shards(values: Iterable[str]) -> list[str]:
    expanded_paths = []
    seen = set()
    for pattern in _split_data_sources(values):
        for path in _expand_one_pattern(pattern):
            resolved = str(Path(path).expanduser())
            if resolved in seen:
                continue
            seen.add(resolved)
            expanded_paths.append(resolved)

    missing = [path for path in expanded_paths if not Path(path).is_file()]
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Input shard paths do not exist:\n{sample}")
    return expanded_paths


def ensure_output_dir_is_ready(output_dir: Path, output_prefix: str) -> None:
    existing_outputs = sorted(output_dir.glob(f"{output_prefix}_*.tar"))
    if existing_outputs:
        sample = "\n".join(str(path) for path in existing_outputs[:10])
        raise ValueError(
            "Output directory already contains matching output shards. Use a fresh directory or a new prefix:\n"
            f"{sample}"
        )


def split_key_and_suffix(member_name: str) -> tuple[str | None, str | None]:
    if not member_name or member_name.endswith("/"):
        return None, None
    prefix, dot, suffix = member_name.rpartition(".")
    if not dot or not prefix or not suffix:
        return None, None
    return prefix, suffix.lower()


def decode_text_value(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def iter_tar_samples(tar_path: str) -> Iterator[tuple[str | None, SampleMembers]]:
    current_key = None
    current_source = None
    current_members: SampleMembers = []

    with tarfile.open(tar_path, "r:*") as src:
        for member in src:
            if not member.isfile():
                continue

            key, suffix = split_key_and_suffix(member.name)
            if key is None:
                continue

            extracted = src.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()

            if current_key is None:
                current_key = key
            elif key != current_key:
                yield current_source, current_members
                current_key = key
                current_source = None
                current_members = []

            current_members.append((member, data))
            if suffix == "source":
                current_source = decode_text_value(data)

    if current_key is not None:
        yield current_source, current_members


def _copy_member(member: tarfile.TarInfo, data: bytes, dst: tarfile.TarFile, name: str | None = None) -> int:
    info = tarfile.TarInfo(name=name or member.name)
    info.size = len(data)
    info.mode = member.mode or 0o644
    info.mtime = member.mtime
    info.type = tarfile.REGTYPE
    dst.addfile(info, io.BytesIO(data))
    return len(data)


def build_temp_output_path(temp_dir: Path, input_index: int, input_path: str) -> str:
    return str(temp_dir / f"filtered_{input_index:06d}_{Path(input_path).name}")


def filter_one_shard(input_path: str, temp_path: str, source: str) -> FilteredShardResult:
    matched_samples = 0
    total_samples = 0
    written_bytes = 0

    with tarfile.open(temp_path, "w") as dst:
        for sample_source, members in iter_tar_samples(input_path):
            total_samples += 1
            if sample_source != source:
                continue
            matched_samples += 1
            for member, data in members:
                written_bytes += _copy_member(member, data, dst)

    if matched_samples == 0:
        Path(temp_path).unlink(missing_ok=True)
        temp_path = None

    return FilteredShardResult(
        input_path=input_path,
        temp_path=temp_path,
        total_samples=total_samples,
        matched_samples=matched_samples,
        written_bytes=written_bytes,
    )


def repack_filtered_shards(
    filtered_results: list[FilteredShardResult],
    output_dir: Path,
    output_prefix: str,
    max_samples_per_shard: int,
) -> DenseShardWriter:
    writer = DenseShardWriter(
        output_dir=output_dir,
        output_prefix=output_prefix,
        max_samples_per_shard=max_samples_per_shard,
    )
    for result in filtered_results:
        if result.temp_path is None:
            continue
        for _, members in iter_tar_samples(result.temp_path):
            writer.add_sample(members)

    writer.close()
    return writer


def write_manifest(output_dir: str, manifest_name: str, shard_paths: list[str]) -> str:
    manifest_path = Path(output_dir) / manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        for shard_path in shard_paths:
            handle.write(f"{shard_path}\n")
    return str(manifest_path)


def write_size_metadata(output_dir: Path, total_samples: int, shard_sample_counts: dict[str, int]) -> tuple[str, str]:
    sizes_path = output_dir / "sizes.json"
    with sizes_path.open("w", encoding="utf-8") as handle:
        json.dump(shard_sample_counts, handle, indent=2, sort_keys=True)
        handle.write("\n")

    len_path = output_dir / "__len__"
    len_path.write_text(str(total_samples), encoding="utf-8")
    return str(sizes_path), str(len_path)


def build_output_brace_expression(output_dir: Path, output_prefix: str, num_shards: int) -> str | None:
    if num_shards <= 0:
        return None
    if num_shards == 1:
        return str(output_dir / f"{output_prefix}_000000.tar")
    return str(output_dir / f"{output_prefix}_{{000000..{num_shards - 1:06d}}}.tar")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    shard_paths = expand_input_shards(args.input_shards)
    if not shard_paths:
        raise RuntimeError("No input shards were found after expansion.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_dir_is_ready(output_dir, args.output_prefix)

    num_workers = max(1, min(args.num_workers, len(shard_paths)))
    LOGGER.info(
        "Filtering %d shards with %d workers for source=%s into max %d samples/shard",
        len(shard_paths),
        num_workers,
        args.source,
        args.max_samples_per_shard,
    )

    total_samples = 0
    matched_samples = 0
    filtered_bytes = 0

    with tempfile.TemporaryDirectory(prefix="filtered_shards_", dir=output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        ordered_results: list[FilteredShardResult | None] = [None] * len(shard_paths)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(
                    filter_one_shard,
                    input_path,
                    build_temp_output_path(temp_dir, index, input_path),
                    args.source,
                ): index
                for index, input_path in enumerate(shard_paths)
            }

            for completed, future in enumerate(concurrent.futures.as_completed(future_to_index), start=1):
                index = future_to_index[future]
                result = future.result()
                ordered_results[index] = result
                total_samples += result.total_samples
                matched_samples += result.matched_samples
                filtered_bytes += result.written_bytes

                if completed % args.log_every == 0 or completed == len(future_to_index):
                    LOGGER.info(
                        "Filtered %d/%d input shards | matched_samples=%d | temp_gb=%.2f",
                        completed,
                        len(future_to_index),
                        matched_samples,
                        filtered_bytes / (1024 ** 3),
                    )

        filtered_results = [result for result in ordered_results if result is not None]
        writer = repack_filtered_shards(
            filtered_results=filtered_results,
            output_dir=output_dir,
            output_prefix=args.output_prefix,
            max_samples_per_shard=args.max_samples_per_shard,
        )

    manifest_path = write_manifest(str(output_dir), args.manifest_name, writer.shard_paths)
    sizes_path, len_path = write_size_metadata(output_dir, writer.total_samples, writer.shard_sample_counts)
    brace_expression = build_output_brace_expression(output_dir, args.output_prefix, len(writer.shard_paths))
    LOGGER.info(
        "Finished filtering | total_samples=%d | matched_samples=%d | output_shards=%d | manifest=%s",
        total_samples,
        matched_samples,
        len(writer.shard_paths),
        manifest_path,
    )
    LOGGER.info(
        "Output dataset metadata | __len__=%s | sizes=%s | written_gb=%.2f",
        len_path,
        sizes_path,
        writer.total_written_bytes / (1024 ** 3),
    )
    if brace_expression is not None:
        LOGGER.info(
            "Output shard expression: %s",
            brace_expression,
        )
    else:
        LOGGER.info("No samples matched source=%s; output shard set is empty.", args.source)


if __name__ == "__main__":
    main()