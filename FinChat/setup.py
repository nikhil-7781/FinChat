from mistralai import Mistral
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import ServerlessSpec
from langchain.memory import ConversationBufferWindowMemory


memory = ConversationBufferWindowMemory(
    k=8,  # last 5 exchanges
    memory_key="chat_history",
    return_messages=True
)


def backend_bot_llm(message):
    load_dotenv()
    api_key=os.getenv('MISTRAL_API')
    model="open-mistral-7b"

    client=Mistral(api_key=api_key)
    model_embed=SentenceTransformer("all-MiniLM-L6-v2")

    pinecone_api=os.getenv('PINECONE_API')
    pc=pinecone.Pinecone(api_key=pinecone_api)



    instruction = f"""
You are an expert in stock market trends. You act as a peer-to-peer mentor who debates ideas with the user, quizzes them to confirm understanding, and explains reasoning clearly.

Rules for every response:

1. **Understand and Confirm:** Before giving an opinion, restate your understanding of the user's question or strategy and ask for confirmation.
2. **Clear Opinion:** Give your opinion as either FOR or AGAINST a statement or strategy. Avoid vague or diplomatic responses.
3. **Explain with Evidence:** Support your view with logical reasoning, historical data, market trends, technical indicators (RSI, MACD, SMA, EMA, etc.), and financial ratios.
4. **Highlight Conflicts and Uncertainty:** 
   - If indicators conflict (e.g., bullish MACD but overbought RSI), explain both sides.
   - Clarify possible risks or scenarios where the expected outcome might change.
5. **Ask a Follow-up Question:** After each opinion, ask exactly one follow-up question to encourage a continuing discussion.
6. **Peer-to-Peer Style:** Speak as a knowledgeable peer, not a formal analyst. Challenge the user’s assumptions politely when needed.
7. **Use Context:** Base your reasoning only on the relevant financial ratios, historical data, static knowledge, or recent market news provided for the conversation.
8. **Avoid Assumptions:** If data is missing (e.g., 1-year change not available), acknowledge it and discuss what that implies.
If multiple indicators give conflicting signals (e.g., RSI suggests overbought while MACD is bullish), explicitly mention the mixed signal before providing your opinion.
Example approach:

- Repeat user's understanding.
- Confirm understanding.
- Give FOR/AGAINST opinion with reasoning.
- Highlight any conflicting signals or uncertainty.
- Always provide opinion, but mention the uncertainties
- Ask one follow-up question.

Always aim for clarity, fairness, and a mentoring conversation style.
"""



    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs=text_splitter.split_text(message)
    embeddings=model_embed.encode(docs, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

    index1=pc.Index("pdf-1-vec-store-v2")
    index2=pc.Index("news-embeddings")
    index3=pc.Index("ticker-ratios")
    chunks=[]

    response1=index1.query(
    namespace="__default__",
    vector=embeddings[0].tolist(),
    top_k=2,
    include_values=False,
    include_metadata=True
)
    for match in response1.get("matches", []):
        if "metadata" in match and "text" in match["metadata"]:
                    chunks.append(match["metadata"]["text"])

    response2=index2.query(
    namespace="__default__",
    vector=embeddings[0].tolist(),
    top_k=2,
    include_values=False,
    include_metadata=True
)
    for match in response2.get("matches", []):
        if "metadata" in match and "text" in match["metadata"]:
                    chunks.append(match["metadata"]["text"])

    response3=index3.query(
    namespace="__default__",
    vector=embeddings[0].tolist(),
    top_k=3,
    include_values=False,
    include_metadata=True
)
    for match in response3.get("matches", []):
        if "metadata" in match and "text" in match["metadata"]:
                    chunks.append(match["metadata"]["text"])
    
    context = "\n".join(chunks)

    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    history_str = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in chat_history]
    )
    augmented_prompt = (
        f"Conversation so far:\n{history_str}\n\n"
        f"User question: {message.strip()}\n\n"
        f"Relevant extracted text from knwoledge base:\n{context}\n\n"
        "Answer the user question using ONLY the relevant extracted text and always consider the conversation context"
    )
    
    chat_response=client.chat.complete(
        model=model,
        messages=[
            {
                "role":"system",
                "content":instruction
            },
            {
                "role":"user",
                "content": augmented_prompt
            },
        
        ],
        max_tokens=700,
        temperature=0.6

    )

    bot_reply=chat_response.choices[0].message.content

    memory.save_context(
                {"input": message},
                {"output": bot_reply}
    )
    return bot_reply