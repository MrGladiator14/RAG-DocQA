from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
import hashlib
import sys


def generate_session_id() -> str:
    """
    Generates a unique session ID incorporating a timestamp and a random hex string.

    Returns:
        str: A unique session ID string (e.g., 'session_20231026_223045_abcdef12').
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


class ChatIngestor:
    """
    Manages the document ingestion process, including file saving, document loading,
    text splitting, and building a FAISS-backed retriever for a chat application.

    It handles session-specific directory management for temporary files and FAISS indices.
    """
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initializes the ChatIngestor, sets up directories, and loads the embedding model.

        Args:
            temp_base (str): Base directory for storing uploaded files temporarily.
                             Defaults to "data".
            faiss_base (str): Base directory for storing FAISS index files.
                              Defaults to "faiss_index".
            use_session_dirs (bool): If True, creates subdirectories under `temp_base`
                                     and `faiss_base` using the `session_id`. Defaults to True.
            session_id (Optional[str]): A predefined session ID. If None, a unique ID is
                                        generated. Defaults to None.

        Raises:
            DocumentPortalException: If there is an error during initialization (e.g.,
                                     failing to load models).
        """
        try:
            self.model_loader = ModelLoader()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                     session_id=self.session_id,
                     temp_dir=str(self.temp_dir),
                     faiss_dir=str(self.faiss_dir),
                     sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e


    def _resolve_dir(self, base: Path) -> Path:
        """
        Resolves the final working directory path based on session setting.

        If `self.use_session` is True, it returns a session-specific subdirectory
        of the base path, creating it if it doesn't exist. Otherwise, it returns
        the base path itself.

        Args:
            base (Path): The base directory path (e.g., self.faiss_base).

        Returns:
            Path: The resolved working directory path.
        """
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/session_..."
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"

    def _split(self, docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Splits a list of LangChain Documents into smaller chunks using a
        RecursiveCharacterTextSplitter.

        Args:
            docs (List[Document]): The list of documents to be split.
            chunk_size (int): The maximum size of each text chunk. Defaults to 1000.
            chunk_overlap (int): The overlap between consecutive chunks. Defaults to 200.

        Returns:
            List[Document]: A list of smaller, chunked documents.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5):
        """
        The main ingestion pipeline. Saves files, loads documents, splits text,
        creates/loads a FAISS index, and returns a configured LangChain Retriever.

        Args:
            uploaded_files (Iterable): An iterable of file-like objects (e.g., from a
                                       web framework's upload handler).
            chunk_size (int): The size of text chunks. Defaults to 1000.
            chunk_overlap (int): The overlap between text chunks. Defaults to 200.
            k (int): The number of documents to return in a search. Defaults to 5.
            search_type (str): The search algorithm to use ('similarity' or 'mmr').
                               Defaults to "mmr".
            fetch_k (int): The number of documents to fetch before applying MMR.
                           Only used if `search_type` is "mmr". Defaults to 20.
            lambda_mult (float): The diversity parameter for MMR. 0 is minimum diversity,
                                 1 is maximum diversity. Only used if `search_type` is "mmr".
                                 Defaults to 0.5.

        Returns:
            langchain.schema.retriever.BaseRetriever: A configured LangChain retriever
                                                      (FAISS-backed).

        Raises:
            ValueError: If no valid documents are loaded from the uploaded files.
            DocumentPortalException: For any failure during the ingestion process.
        """
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            try:
                # Attempt to load or create the index. The double-try is a fail-safe
                # for potential race conditions or initial creation issues.
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)

            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            # Configure search parameters based on search type
            search_kwargs = {"k": k}
            
            if search_type == "mmr":
                # MMR needs fetch_k (docs to fetch) and lambda_mult (diversity parameter)
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
            
            return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

class FaissManager:
    """
    Handles the loading, creation, and persistence of a FAISS vector store.

    It manages a metadata file to ensure idempotent document addition (avoiding
    re-ingestion of the same document chunks).
    """
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        """
        Initializes the FaissManager.

        Args:
            index_dir (Path): The directory where the FAISS index files (index.faiss,
                              index.pkl) and metadata file (ingested_meta.json) are stored.
            model_loader (Optional[ModelLoader]): An instance of ModelLoader to load the
                                                  embedding model. If None, a new one is
                                                  instantiated. Defaults to None.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                # load it if already there
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                # init the empty one if it fails to load
                self._meta = {"rows": {}}


        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self)-> bool:
        """
        Checks if the FAISS index files already exist in the index directory.

        Returns:
            bool: True if both 'index.faiss' and 'index.pkl' exist, False otherwise.
        """
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Generates a unique key for a document chunk to check for idempotency.

        Prioritizes a combination of 'source'/'file_path' and 'row_id' from metadata.
        Falls back to a SHA256 hash of the text content if source info is missing.

        Args:
            text (str): The page content of the document chunk.
            md (Dict[str, Any]): The metadata dictionary of the document chunk.

        Returns:
            str: The unique fingerprint key.
        """
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        """
        Saves the current ingestion metadata (self._meta) to the 'ingested_meta.json' file.
        """
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")


    def add_documents(self,docs: List[Document]) -> int:
        """
        Adds a list of LangChain Documents (chunks) to the internal FAISS index
        in an idempotent manner. Only documents whose fingerprint is not found in
        the metadata are added.

        The index is saved to disk after adding new documents.

        Args:
            docs (List[Document]): The list of documents to add.

        Returns:
            int: The number of new documents added to the FAISS index.

        Raises:
            RuntimeError: If `load_or_create()` has not been called successfully
                          to initialize `self.vs`.
        """
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []

        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None) -> FAISS:
        """
        Loads an existing FAISS vector store from disk if index files exist.
        If not, it creates a new FAISS index from the provided texts and metadatas,
        and then saves it to disk.

        Args:
            texts (Optional[List[str]]): List of text content for creating a new index.
                                        Required if no index exists. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadata dictionaries for creating a
                                              new index. Defaults to None.

        Returns:
            FAISS: The loaded or newly created FAISS vector store instance.

        Raises:
            DocumentPortalException: If no existing index is found and no `texts` are
                                     provided to create one.
        """
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
            
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs