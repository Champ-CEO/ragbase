import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from ragbase.config import Config
from ragbase.optimizers import DynamicComplexityRouter, TokenOptimizationPipeline
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question.
If the answer is not found within the context, state that the answer cannot be found.
Prioritize concise responses (maximum of 3 sentences) and use a list where applicable.
The contextual information is organized with the most relevant source appearing first.
Each source is separated by a horizontal rule (---).

Context:
{context}

Use markdown formatting where appropriate.
"""


def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")

    return remove_links("\n".join(texts))


def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name": "context_retriever"})
            | format_documents
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "chain_answer"})


def update_chain_llm(chain: Runnable, llm: BaseLanguageModel) -> Runnable:
    """Update the LLM in an existing chain."""
    # For RunnableWithMessageHistory, we need to access the inner chain
    if hasattr(chain, "runnable"):
        # If it's a RunnableWithMessageHistory, extract the inner runnable
        inner_chain = chain.runnable
        
        # Create a new inner chain with the new LLM but same retriever
        # We need to extract the retriever from the current chain
        retriever = None
        # Try to find the retriever in the chain structure
        if hasattr(inner_chain, "retriever"):
            retriever = inner_chain.retriever
        # Sometimes the retriever might be nested in a RunnablePassthrough assign structure
        elif hasattr(inner_chain, "first") and hasattr(inner_chain.first, "retriever"):
            retriever = inner_chain.first.retriever
        
        # If we can't find the retriever, the structure might have changed, so we just
        # create a new chain but keep the message history wrapper
        if retriever is None:
            return chain  # Return original chain, can't safely update it
            
        # Create a new inner chain with the new LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        updated_inner_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question")
                | retriever.with_config({"run_name": "context_retriever"})
                | format_documents
            )
            | prompt
            | llm
        )
        
        # Wrap the new inner chain with the same message history configuration
        return RunnableWithMessageHistory(
            updated_inner_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        ).with_config({"run_name": "chain_answer"})
    else:
        # For simpler chain structures, try direct replacement
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        
        # Get the retriever from the original chain if possible
        retriever = None
        if hasattr(chain, "retriever"):
            retriever = chain.retriever
        
        if retriever is None:
            # If we can't extract the retriever, return the original chain
            return chain
            
        updated_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question")
                | retriever.with_config({"run_name": "context_retriever"})
                | format_documents
            )
            | prompt
            | llm
        )
        
        return RunnableWithMessageHistory(
            updated_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        ).with_config({"run_name": "chain_answer"})


async def ask_question(chain: Runnable, question: str, session_id: str):
    # Apply dynamic complexity routing if enabled
    if Config.AUTO_COMPLEXITY_ROUTING:
        try:
            is_complex, complexity_score = DynamicComplexityRouter.assess_complexity(question)
            # Get the appropriate LLM
            llm, _ = DynamicComplexityRouter.get_appropriate_llm(question)
            # Update the chain with the selected LLM
            updated_chain = update_chain_llm(chain, llm)
            # Only use the updated chain if it was successfully updated
            if updated_chain is not chain:  # This means update was successful
                chain = updated_chain
            else:
                # Log that we're falling back to the original chain
                print("Warning: Could not update chain with dynamic routing, using original chain")
        except Exception as e:
            # If anything goes wrong with dynamic routing, fall back to the original chain
            print(f"Error in dynamic complexity routing: {e}. Using original chain.")
    
    # Apply token optimization if enabled
    if Config.USE_TOKEN_OPTIMIZATION:
        try:
            optimized_question = TokenOptimizationPipeline.optimize_prompt(question)
            # Only use the optimized question if it's not empty
            if optimized_question:
                question = optimized_question
        except Exception as e:
            # If token optimization fails, use the original question
            print(f"Error in token optimization: {e}. Using original question.")
    
    try:
        async for event in chain.astream_events(
            {"question": question},
            config={
                "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
                "configurable": {"session_id": session_id},
            },
            version="v2",
            include_names=["context_retriever", "chain_answer"],
        ):
            event_type = event["event"]
            if event_type == "on_retriever_end":
                yield event["data"]["output"]
            if event_type == "on_chain_stream":
                yield event["data"]["chunk"].content
    except Exception as e:
        # Handle any exceptions during chain execution
        error_message = f"An error occurred while processing your question: {str(e)}"
        print(f"Chain execution error: {e}")
        yield error_message
