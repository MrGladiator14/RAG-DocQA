from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Any
from multi_doc_chat.logger.cutom_logger import CustomLogger
from multi_doc_chat.exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}

# Local logger instance
log = CustomLogger().get_logger(__name__)


def save_uploaded_files(uploaded_files: Iterable[Any], target_dir: Path) -> List[Path]:
    """
    Saves an iterable of uploaded file-like objects to a specified directory.

    This function is designed to handle various file-like objects commonly found
    in web frameworks (e.g., Streamlit's `UploadedFile`, Starlette's `UploadFile`).
    It validates file extensions against `SUPPORTED_EXTENSIONS`, generates a unique
    and safe filename, and writes the file content to the target directory.

    Args:
        uploaded_files: An iterable of file-like objects. Each object is expected
                        to expose either a `.filename` (or `.name`) attribute and
                        a file-like interface (e.g., `.file.read()`, `.read()`, or
                        `.getbuffer()`).
        target_dir: The `pathlib.Path` object representing the directory where the
                    files should be saved. The directory will be created if it
                    does not exist.

    Returns:
        A list of `pathlib.Path` objects, where each path points to one of the
        successfully saved local files.

    Raises:
        DocumentPortalException: If an error occurs during the saving process,
                                 such as an inability to write to the directory
                                 or an unsupported file object interface.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            # Determine original file name and extension
            name = getattr(uf, "filename", getattr(uf, "name", "file"))
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue
            
            # Create a unique filename: UUID hex (8 chars) + extension
            fname = f"{uuid.uuid4().hex[:8]}{ext}"
            out = target_dir / fname
            
            with open(out, "wb") as f:
                # Prefer underlying file buffer when available (e.g., Starlette UploadFile.file)
                if hasattr(uf, "file") and hasattr(uf.file, "read"):
                    f.write(uf.file.read())
                elif hasattr(uf, "read"):
                    data = uf.read()
                    # If a memoryview is returned, convert to bytes; otherwise assume bytes
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    f.write(data)
                else:
                    # Fallback for objects exposing a getbuffer()
                    buf = getattr(uf, "getbuffer", None)
                    if callable(buf):
                        data = buf()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        f.write(data)
                    else:
                        raise ValueError("Unsupported uploaded file object; no readable interface")
            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e