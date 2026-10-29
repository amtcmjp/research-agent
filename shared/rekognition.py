"""AWS Rekognition thin wrapper for ID / face OSINT verification."""

from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

QualityLabel = Literal["high", "medium", "low", "unknown"]


class FaceDetection(TypedDict):
    detected: bool
    count: int
    confidence: float
    quality_label: QualityLabel
    brightness: float
    sharpness: float
    sunglasses: bool
    error: str


class FaceCompareResult(TypedDict):
    compared: bool
    matched: bool
    similarity: float
    matched_count: int
    unmatched_count: int
    error: str


def _rekognition_client() -> Any:
    return boto3.client(
        "rekognition",
        region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
    )


def _download_image(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _label_quality(brightness: float, sharpness: float) -> QualityLabel:
    if brightness >= 60 and sharpness >= 60:
        return "high"
    if brightness >= 30 and sharpness >= 30:
        return "medium"
    return "low"


def detect_face(image_url: str) -> FaceDetection:
    """Run Rekognition DetectFaces against the image at `image_url`."""
    result: FaceDetection = {
        "detected": False,
        "count": 0,
        "confidence": 0.0,
        "quality_label": "unknown",
        "brightness": 0.0,
        "sharpness": 0.0,
        "sunglasses": False,
        "error": "",
    }

    try:
        image_bytes = _download_image(image_url)
    except Exception as exc:
        result["error"] = f"Image download failed: {type(exc).__name__}: {exc}"
        return result

    try:
        response = _rekognition_client().detect_faces(
            Image={"Bytes": image_bytes},
            Attributes=["ALL"],
        )
    except Exception as exc:
        result["error"] = f"Rekognition detect_faces failed: {type(exc).__name__}: {exc}"
        return result

    faces = response.get("FaceDetails") or []
    if not faces:
        return result

    best = max(faces, key=lambda f: f.get("Confidence", 0.0))
    quality = best.get("Quality") or {}
    brightness = float(quality.get("Brightness", 0.0))
    sharpness = float(quality.get("Sharpness", 0.0))

    result.update(
        {
            "detected": True,
            "count": len(faces),
            "confidence": float(best.get("Confidence", 0.0)),
            "quality_label": _label_quality(brightness, sharpness),
            "brightness": brightness,
            "sharpness": sharpness,
            "sunglasses": bool((best.get("Sunglasses") or {}).get("Value", False)),
        }
    )
    return result


def compare_faces(
    source_image_url: str,
    target_image_url: str,
    similarity_threshold: float = 70.0,
) -> FaceCompareResult:
    """Compare two face images with Rekognition CompareFaces."""
    result: FaceCompareResult = {
        "compared": False,
        "matched": False,
        "similarity": 0.0,
        "matched_count": 0,
        "unmatched_count": 0,
        "error": "",
    }

    try:
        source_bytes = _download_image(source_image_url)
        target_bytes = _download_image(target_image_url)
    except Exception as exc:
        result["error"] = f"Image download failed: {type(exc).__name__}: {exc}"
        return result

    try:
        response = _rekognition_client().compare_faces(
            SourceImage={"Bytes": source_bytes},
            TargetImage={"Bytes": target_bytes},
            SimilarityThreshold=similarity_threshold,
            QualityFilter="AUTO",
        )
    except Exception as exc:
        result["error"] = f"Rekognition compare_faces failed: {type(exc).__name__}: {exc}"
        return result

    matches = response.get("FaceMatches") or []
    unmatched = response.get("UnmatchedFaces") or []
    best_similarity = max((m.get("Similarity", 0.0) for m in matches), default=0.0)

    result.update(
        {
            "compared": True,
            "matched": bool(matches),
            "similarity": float(best_similarity),
            "matched_count": len(matches),
            "unmatched_count": len(unmatched),
        }
    )
    return result
