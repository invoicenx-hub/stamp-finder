"""
Stamp/Signature Location Finder API
------------------------------------
Upload a PDF (e.g. Nixon Cleaning quotations) and get back the page number
and coordinates of the actual "簽署及蓋章" (sign & stamp) block — filtering
out the false-positive that appears inside the boilerplate sentence
("如對上述報價無異議，請簽署及蓋章寄回作實").

Deploy on Hugging Face Spaces using the "FastAPI" / Docker template.
"""

import io
import tempfile
from typing import List, Dict, Any

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stamp Finder API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keywords that mark a genuine sign/stamp block
TARGET_PHRASE = "簽署及蓋章"
# Words that indicate the match is part of the boilerplate sentence, not the
# actual signature cell (checked on the same text line)
CLAUSE_MARKERS = ["請", "寄回作實", "回覆作實", "如對上述報價"]
# A genuine signature cell is almost always followed shortly (same or next
# line, similar x-position) by a "日期" (date) label.
DATE_MARKER = "日期"


def _line_key(word: Dict[str, Any], tolerance: float = 3.0) -> float:
    """Round the 'top' coordinate so words on the same visual line group together."""
    return round(word["top"] / tolerance) * tolerance


def find_stamp_candidates(pdf_bytes: bytes) -> Dict[str, Any]:
    result = {"page_count": 0, "candidates": []}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        result["page_count"] = len(pdf.pages)

        for page_index, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False)
            if not words:
                continue

            # Group words into lines by rounded 'top' coordinate
            lines: Dict[float, List[Dict[str, Any]]] = {}
            for w in words:
                lines.setdefault(_line_key(w), []).append(w)

            sorted_line_keys = sorted(lines.keys())

            # Try to find the target phrase either as one token or split
            # across consecutive word tokens on the same line.
            full_text_per_line = {
                k: "".join(w["text"] for w in sorted(v, key=lambda x: x["x0"]))
                for k, v in lines.items()
            }

            for k in sorted_line_keys:
                line_text = full_text_per_line[k]
                if TARGET_PHRASE not in line_text:
                    continue

                # Reject if this line is clearly the boilerplate clause
                if any(marker in line_text for marker in CLAUSE_MARKERS):
                    continue

                # Get approx coordinates of the phrase on this line
                line_words = sorted(lines[k], key=lambda x: x["x0"])
                x0 = min(w["x0"] for w in line_words)
                top = min(w["top"] for w in line_words)

                # Look for a nearby "日期" label within the next ~40pt below,
                # roughly same horizontal region (confirms it's a sign-off block)
                has_date_nearby = False
                for k2 in sorted_line_keys:
                    if k2 <= k or k2 - k > 40:
                        continue
                    if DATE_MARKER in full_text_per_line[k2]:
                        # roughly same column
                        other_words = lines[k2]
                        other_x0 = min(w["x0"] for w in other_words)
                        if abs(other_x0 - x0) < 150:
                            has_date_nearby = True
                            break

                result["candidates"].append(
                    {
                        "page": page_index,
                        "x0": round(x0, 1),
                        "top": round(top, 1),
                        "page_width": round(page.width, 1),
                        "page_height": round(page.height, 1),
                        "confidence": "high" if has_date_nearby else "medium",
                        "line_text": line_text,
                    }
                )

    # Sort so the highest-confidence, latest-page candidate comes first
    result["candidates"].sort(
        key=lambda c: (c["confidence"] != "high", -c["page"])
    )
    return result


@app.get("/")
def root():
    return {
        "status": "ok",
        "usage": "POST a PDF file to /find-stamp as multipart/form-data field 'file'",
    }


@app.post("/find-stamp")
async def find_stamp(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    pdf_bytes = await file.read()

    try:
        result = find_stamp_candidates(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    if not result["candidates"]:
        result["message"] = (
            "No clear sign/stamp block found. The PDF may be scanned (image-only) "
            "rather than text-based — OCR would be needed."
        )

    return result
