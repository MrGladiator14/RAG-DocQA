from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at handling conversational flow. Your task is to analyze the conversation history "
        "and the latest user message. **If the latest message is context-dependent, rewrite it as a "
        "fully independent question** that can be understood by itself. If it's already a complete question, "
        "return it as is. Your output must only be the revised or original question, nothing else."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a factual, context-aware answering bot. Your primary rule is to **strictly use the provided "
        "context to formulate your answer.** If the information necessary to answer the user's question is "
        "not present in the context, you must reply with the phrase: 'The provided context does not contain this information.' "
        "Ensure your response is clear, direct, and limited to a maximum of two sentences.\n\nContext:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}