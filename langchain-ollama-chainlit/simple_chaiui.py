from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(base_url="http://localhost:11434", model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente basado en inteligencia artificial que ayuda a los usuarios que le preguntan respondiendo siempre de forma correcta y educada. Si no sabe algo, simplemente responde No tengo información sobre su pregunta. Puedes responder cualquier pregunta relacioanda con informática. Responde siempre en Español."
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
