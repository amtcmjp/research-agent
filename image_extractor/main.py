from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Final

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE: Final[int] = 5 * 1024 * 1024

SUPPORTED_IMAGE_TYPES: Final[dict[str, str]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class TextExtract(BaseModel):
    """Personal information extracted from the front and back images."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description=(
            "The person's name exactly as visibly printed on either image. "
            "Return null if missing or unreadable."
        ),
    )

    address: str | None = Field(
        default=None,
        description=(
            "The complete address visibly printed on either image. "
            "Preserve the original characters and address notation. "
            "Return null if missing or unreadable."
        ),
    )


EXTRACTION_PROMPT: Final[str] = """
You are extracting information from the FRONT and BACK images
of the same card.

Image 1 is the FRONT.
Image 2 is the BACK.

Extract only these fields:
- name
- address

Rules:
1. Examine both images before producing the result.
2. Use information visible in either image.
3. Preserve Japanese kanji, hiragana, katakana, Latin characters,
   punctuation, and address notation exactly as printed.
4. Do not translate, guess, correct, or complete missing information.
5. Do not mistake an organization name, label, or other text
   for the person's name.
6. If the same field appears in both images, return it only once.
7. If an address is split across visible lines, combine the lines
   only when the printed text clearly belongs to the same address.
8. If a field is missing or unreadable in both images, return null.
9. Do not infer personal information from outside knowledge.
10. Return only the fields defined in the output schema.
"""


def create_llm() -> ChatBedrockConverse:
    """Create a Bedrock vision model."""

    region = os.getenv("AWS_REGION", "ap-northeast-1").strip()
    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
    ).strip()

    if not region or not model_id:
        raise ValueError("AWS region and model ID must not be empty.")

    return ChatBedrockConverse(
        model=model_id,
        region_name=region,
        temperature=0,
        max_tokens=1024,
    )


def load_image(image_path: Path) -> tuple[str, str]:
    """Validate an image and return its MIME type and base64 data."""

    if not image_path.is_file():
        raise FileNotFoundError("An image file was not found.")

    if image_path.stat().st_size == 0:
        raise ValueError("An image file is empty.")

    if image_path.stat().st_size > MAX_IMAGE_SIZE:
        raise ValueError("Each image must be no larger than 5 MiB.")

    try:
        with Image.open(image_path) as image:
            image_format = image.format

            if image_format not in SUPPORTED_IMAGE_TYPES:
                raise ValueError("Only JPEG, PNG, and WebP images are supported.")

            image.verify()

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("An image is invalid or unreadable.") from exc

    image_bytes = image_path.read_bytes()
    mime_type = SUPPORTED_IMAGE_TYPES[image_format]
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    return mime_type, image_base64


def extract_text(front_path: Path, back_path: Path) -> TextExtract:
    """Extract a person's name and address from both card images."""

    logger.info("Validating front and back images.")

    front_mime, front_base64 = load_image(front_path)
    back_mime, back_base64 = load_image(back_path)

    logger.info("Sending both images to the vision model.")
    structured_llm = create_llm().with_structured_output(TextExtract)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": EXTRACTION_PROMPT,
            },
            {
                "type": "text",
                "text": "Image 1: FRONT of the card.",
            },
            {
                "type": "image",
                "base64": front_base64,
                "mime_type": front_mime,
            },
            {
                "type": "text",
                "text": "Image 2: BACK of the same card.",
            },
            {
                "type": "image",
                "base64": back_base64,
                "mime_type": back_mime,
            },
        ]
    )

    result = structured_llm.invoke([message])

    if isinstance(result, TextExtract):
        return result

    return TextExtract.model_validate(result)


def main() -> int:
    """Run extraction from the command line."""

    front_image = Path("/Users/aungmyotun/Downloads/01.jpg")
    back_image = Path("/Users/aungmyotun/Downloads/01.jpg")
    try:
        result = extract_text(front_image, back_image)
        output = result.model_dump_json(indent=2)
        print(output)

        return 0

    except FileNotFoundError:
        logger.error("An image file was not found.")
    except ValueError as exc:
        logger.error("Input or validation error: %s", exc)
    except (ClientError, BotoCoreError) as exc:
        logger.error("AWS request failed: %s", type(exc).__name__)
    except Exception as exc:
        logger.error("Extraction failed: %s", type(exc).__name__)

    return 1


if __name__ == "__main__":
    sys.exit(main())
