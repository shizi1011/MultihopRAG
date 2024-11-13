import os
import chainlit as cl
from dotenv import load_dotenv
load_dotenv('.env')
from ingest import file_ingestor, create_vector_db
from model import LLMLoader, EmbeddingLoader, MultihopRAGSystem

@cl.on_chat_start
async def start():
    embeddings = EmbeddingLoader.from_huggingface_embedding()
    llm = LLMLoader.from_openai(openai_api_key=os.getenv("OPENAI_API_KEY"))

    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Hi, Welcome to RAG Chatbot. Please upload a PDF file to get started !", accept=["application/pdf"]
        ).send()
    
    text_file = files[0]

    await cl.Message(
        content=f"`{text_file.name}` uploaded!"
    ).send()

    msg = cl.Message(content="Indexing file...")
    await msg.send()

    docs = file_ingestor(text_file.path)
    vector_db = await create_vector_db(docs, embeddings)
    rag = MultihopRAGSystem(
        retriever=vector_db.as_retriever(),
        llm=llm
    )



    # Let the user know that the system is ready
    msg.content = "File indexed successfully!, please ask a question"
    await msg.update()

    cl.user_session.set("rag", rag)

@cl.on_message
async def main(message: cl.Message):
    rag = cl.user_session.get("rag") 
    # cb = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True
    #     # , answer_prefix_tokens=["Final", "Answer"]
    # )
    # cb.answer_reached = True
    # res = await rag.ainvoke(message.content, callbacks=[cb])
    answer, context= await rag.forward(message.content)

    # answer += f"\nSources:" + str(context)

    # await cl.Message(content=answer).send()
    
    text_elements = []  
    for i, ctx in enumerate(context):
        text_elements.append(
            cl.Text(content=ctx, name=f'context {i}', display="side")
        )
    source_names = [text_el.name for text_el in text_elements]
    answer += f"\nSources: {', '.join(source_names)}"
    await cl.Message(content=answer, elements=text_elements).send()
    
