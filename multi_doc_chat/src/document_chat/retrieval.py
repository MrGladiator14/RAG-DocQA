import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError


class ConversationalRAG:
    """
    An LCEL-based Conversational RAG (Retrieval-Augmented Generation) system
    that supports lazy initialization of the retriever.

    This class manages the entire RAG pipeline, including conversational
    contextualization of the user's question, document retrieval, and
    final answer generation using an LLM.

    Usage:
        >>> rag = ConversationalRAG(session_id="abc-123")
        >>> rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5)
        >>> answer = rag.invoke("What is the main topic of the document?", chat_history=[])
        'The main topic is...'
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        """
        Initializes the ConversationalRAG instance.

        Loads the Language Model (LLM) and necessary prompt templates.
        The RAG chain is built immediately if a retriever is provided.

        Args:
            session_id (Optional[str]): A unique identifier for the chat session,
                                        used for logging.
            retriever (Optional): An initial LangChain-compatible retriever object.
        
        Raises:
            DocumentPortalException: If initialization fails (e.g., model loading error).
        """
        try:
            self.session_id = session_id

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Loads a FAISS vector store from disk, creates a retriever from it,
        and builds the full LCEL RAG chain.

        Args:
            index_path (str): The file path to the FAISS index directory.
            k (int): The number of retrieved documents to return to the LLM. Defaults to 5.
            index_name (str): The name of the index file within `index_path`. Defaults to "index".
            search_type (str): The type of search to perform. Options include
                               "similarity", "mmr" (Maximum Marginal Relevance), or
                               "similarity_score_threshold". Defaults to "mmr".
            fetch_k (int): Number of documents to fetch before MMR re-ranking (only used for "mmr" search).
                           Defaults to 20.
            lambda_mult (float): Diversity parameter for MMR (0.0 = max diversity, 1.0 = max relevance).
                                 Defaults to 0.5.
            search_kwargs (Optional[Dict[str, Any]]): Custom search keyword arguments
                                                     that will override other search parameters
                                                     if provided.

        Returns:
            The configured LangChain retriever object.

        Raises:
            FileNotFoundError: If the `index_path` directory does not exist.
            DocumentPortalException: If loading the retriever or building the chain fails.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                search_type=search_type,
                k=k,
                fetch_k=fetch_k if search_type == "mmr" else None,
                lambda_mult=lambda_mult if search_type == "mmr" else None,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Executes the full LCEL RAG pipeline to generate a response.

        The pipeline:
        1. Contextualizes `user_input` using `chat_history`.
        2. Retrieves relevant documents based on the contextualized question.
        3. Generates a final answer using the LLM, retrieved documents, and chat history.

        Args:
            user_input (str): The user's current question or statement.
            chat_history (Optional[List[BaseMessage]]): The history of the conversation
                                                        as a list of LangChain BaseMessage objects.
                                                        Defaults to an empty list.

        Returns:
            str: The final, generated answer from the LLM.

        Raises:
            DocumentPortalException: If the RAG chain has not been initialized (retriever not loaded)
                                      or if invocation fails.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            # Validate answer type and length using Pydantic model
            try:
                validated = ChatAnswer(answer=str(answer))
                answer = validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        """
        Loads the Language Model (LLM) using the ModelLoader utility.

        Returns:
            The loaded LangChain-compatible LLM object.

        Raises:
            DocumentPortalException: If the LLM fails to load.
        """
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        """
        Formats a list of LangChain documents into a single string for context injection.

        Args:
            docs (List[Document]): A list of LangChain Document objects.

        Returns:
            str: A single string where document content is joined by double newlines.
        """
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        """
        Constructs the LangChain Expression Language (LCEL) graph for the RAG pipeline.

        The chain consists of three main steps:
        1. Contextualize the question based on chat history.
        2. Retrieve documents using the contextualized question.
        3. Generate the final answer using the original input, history, and retrieved context.

        Raises:
            DocumentPortalException: If no retriever is set or if the chain construction fails.
        """
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)