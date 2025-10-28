from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Loads documents from the specified file paths using the appropriate LangChain loader
    based on the file extension.

    Only files with extensions in `SUPPORTED_EXTENSIONS` (currently .pdf, .docx, .txt)
    are processed. Unsupported files are logged and skipped.

    Args:
        paths (Iterable[Path]): An iterable of `pathlib.Path` objects pointing to the
                                 documents to be loaded.

    Returns:
        List[Document]: A list of LangChain `Document` objects containing the loaded content.

    Raises:
        DocumentPortalException: If an error occurs during the document loading process.
    """
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e


class FastAPIFileAdapter:
    """
    Adapts a `fastapi.UploadFile` object to an interface suitable for mock file handling
    or compatibility layers.

    It exposes the file name via the `.name` attribute and provides a `.getbuffer()`
    method to retrieve the file content as bytes, similar to standard file-like objects
    or in-memory file representations. This is useful for passing FastAPI files to
    libraries expecting a simpler file structure.
    """
    def __init__(self, uf: UploadFile):
        """
        Initializes the adapter with a FastAPI UploadFile.

        Args:
            uf (UploadFile): The FastAPI UploadFile instance.
        """
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        """
        Reads and returns the content of the uploaded file as a bytes object.

        Note: The underlying file pointer is reset to the beginning (seek(0)) before
        reading to ensure the entire content is captured.

        Returns:
            bytes: The content of the uploaded file.
        """
        self._uf.file.seek(0)
        return self._uf.file.read()