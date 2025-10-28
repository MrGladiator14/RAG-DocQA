import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


def test_document_ingestion_and_rag():
    """
    Runs an end-to-end interactive test of the multi-document RAG chat system.

    This script performs the following steps:
    1.  Specifies one or more local PDF/document files for testing.
    2.  Initializes a `ChatIngestor` to create a new session.
    3.  Invokes the `build_retriever` method to process the documents:
        - Documents are loaded, split into chunks, and vectorized.
        - A FAISS index is created and saved to a session-specific directory.
    4.  Initializes a `ConversationalRAG` pipeline.
    5.  Loads the newly created FAISS index into the RAG pipeline.
    6.  Enters an interactive loop where the user can ask questions.
    7.  The RAG system retrieves relevant context and generates answers.
    """
    try:
        test_file_paths = [
            Path("data") / "2406.08394v3.pdf",
            Path("data") / "2510.14528v2.pdf", # Example of a second file
        ]

        # Validate all file paths
        for path in test_file_paths:
            if not path.exists():
                print(f"File does not exist: {path}")
                sys.exit(1)

        opened_files = [open(path, "rb") for path in test_file_paths]
        try:
            ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)

            ci.built_retriver(
                opened_files,
                chunk_size=200,
                chunk_overlap=20,
                k=5,
                search_type="mmr",
                fetch_k=20,
                lambda_mult=0.5
            )

            session_id = ci.session_id
            index_dir = os.path.join("faiss_index", session_id)

            rag = ConversationalRAG(session_id=session_id)
            rag.load_retriever_from_faiss(
                index_path=index_dir,
                k=5,
                index_name=os.getenv("FAISS_INDEX_NAME", "index"),
                search_type="mmr",
                fetch_k=20,
                lambda_mult=0.5
            )

            chat_history = []
            print("\nType 'exit' to quit the chat.\n")
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting chat.")
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q", ":q"}:
                    print("Goodbye!")
                    break

                answer = rag.invoke(user_input, chat_history=chat_history)
                print("Assistant:", answer)

                # Maintain conversation history
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=answer))

        finally:
            for f in opened_files:
                f.close()

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_document_ingestion_and_rag()