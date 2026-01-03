from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from backend.schemas.models import (
    ChunkMarkerRequest,
    ChunkMarkerResponse,
    ChunkPreviewRequest,
    ChunkPreviewResponse,
    ChunkSnapshotRequest,
    ChunkSnapshotResponse,
    UIStateChunkPreview,
)
from text_tools import (
    ChunkInfo,
    MANUAL_CHUNK_MARKER,
    SpeechSegment,
    count_words,
    estimate_duration,
    normalize_text,
    parse_manual_chunks,
    render_clean_text,
    render_clean_text_from_segments,
)


router = APIRouter(prefix="/v1")


def _single_chunk(text: str) -> List[ChunkInfo]:
    clean = render_clean_text(text).strip()
    if not clean:
        return []
    sentence_count = sum(1 for ch in clean if ch in ".!?")
    return [
        ChunkInfo(
            segments=[SpeechSegment("text", clean)],
            sentence_count=sentence_count,
            char_count=len(clean),
            word_count=count_words(clean),
            comma_count=clean.count(","),
            estimated_duration=estimate_duration(clean),
            reason="single",
            boundary_kind="single",
            pivot=False,
            ends_with_suspended=clean.rstrip().endswith((",", ";", ":")),
            oversize_sentence=False,
            warnings=[],
        )
    ]


def _marker_indices(snapshot_text: str) -> List[int]:
    indices = []
    start = 0
    marker = MANUAL_CHUNK_MARKER
    while True:
        idx = snapshot_text.find(marker, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + len(marker)
    return indices


def _insert_marker(snapshot_text: str, position: int) -> str:
    marker = MANUAL_CHUNK_MARKER
    position = max(0, min(position, len(snapshot_text)))
    return f"{snapshot_text[:position]}\n{marker}\n{snapshot_text[position:]}"


def _remove_marker(snapshot_text: str, position: int) -> str:
    marker = MANUAL_CHUNK_MARKER
    indices = _marker_indices(snapshot_text)
    if not indices:
        return snapshot_text
    target = min(indices, key=lambda idx: abs(idx - position))
    start = target
    end = target + len(marker)
    if start > 0 and snapshot_text[start - 1] == "\n":
        start -= 1
    if end < len(snapshot_text) and snapshot_text[end : end + 1] == "\n":
        end += 1
    return snapshot_text[:start] + snapshot_text[end:]


def _chunks_from_ranges(snapshot_text: str, ranges) -> List[ChunkInfo]:
    chunks: List[ChunkInfo] = []
    for entry in ranges:
        if entry.start < 0 or entry.end > len(snapshot_text) or entry.start >= entry.end:
            raise HTTPException(status_code=400, detail="invalid_chunk_range")
        chunk_text = snapshot_text[entry.start : entry.end]
        clean = render_clean_text(chunk_text)
        sentence_count = sum(1 for ch in clean if ch in ".!?")
        chunks.append(
            ChunkInfo(
                segments=[SpeechSegment("text", chunk_text)],
                sentence_count=sentence_count,
                char_count=len(chunk_text),
                word_count=count_words(clean),
                comma_count=clean.count(","),
                estimated_duration=estimate_duration(clean),
                reason="manual_range",
                boundary_kind="manual_range",
                pivot=False,
                ends_with_suspended=clean.rstrip().endswith((",", ";", ":")),
                oversize_sentence=False,
                warnings=[],
            )
        )
    return chunks


def _preview_payload(chunks: List[ChunkInfo]) -> List[UIStateChunkPreview]:
    preview = []
    for idx, chunk in enumerate(chunks, start=1):
        preview.append(
            UIStateChunkPreview(
                index=idx,
                text=render_clean_text_from_segments(chunk.segments).strip(),
                est_duration_s=float(chunk.estimated_duration),
                word_count=int(chunk.word_count),
            )
        )
    return preview


@router.post("/chunks/snapshot", response_model=ChunkSnapshotResponse)
def snapshot_chunks(request: ChunkSnapshotRequest) -> ChunkSnapshotResponse:
    source = request.text_interpreted if request.text_interpreted is not None else request.text_adjusted
    snapshot_text = normalize_text(source or "")
    return ChunkSnapshotResponse(snapshot_text=snapshot_text)


@router.post("/chunks/preview", response_model=ChunkPreviewResponse)
def preview_chunks(request: ChunkPreviewRequest) -> ChunkPreviewResponse:
    snapshot_text = request.snapshot_text or ""
    if request.markers:
        for pos in sorted(set(request.markers), reverse=True):
            snapshot_text = _insert_marker(snapshot_text, int(pos))
    if request.ranges:
        chunks = _chunks_from_ranges(snapshot_text, request.ranges)
        return ChunkPreviewResponse(chunks=_preview_payload(chunks))
    chunks, _marker_count = parse_manual_chunks(snapshot_text, marker=MANUAL_CHUNK_MARKER)
    if not chunks:
        chunks = _single_chunk(snapshot_text)
    return ChunkPreviewResponse(chunks=_preview_payload(chunks))


@router.post("/chunks/apply_marker", response_model=ChunkMarkerResponse)
def apply_marker(request: ChunkMarkerRequest) -> ChunkMarkerResponse:
    if request.action == "insert":
        updated = _insert_marker(request.snapshot_text or "", int(request.position))
    elif request.action == "remove":
        updated = _remove_marker(request.snapshot_text or "", int(request.position))
    else:
        raise HTTPException(status_code=400, detail="invalid_action")
    markers = _marker_indices(updated)
    return ChunkMarkerResponse(snapshot_text_updated=updated, markers_updated=markers)
