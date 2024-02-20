
from dotenv import load_dotenv
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.chat_models.fireworks import ChatFireworks
import os
# os.environ['OPENAI_API_KEY'] = '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'
# os.environ['FIREWORKS_API_KEY'] = "RcFDJALUzuBlUjjCagS6yVECPzsZB8zgLG9Z25tGXHj9e7y5"
# model_path = 'PATH_TO_/models/llama-2-7b-chat.Q4_K_M.gguf'
MODEL_ID = "accounts/fireworks/models/mistral-7b-instruct-4k"
# MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


load_dotenv('.env')


class Loadllm:
    @staticmethod
    def load_llm():

        # Create Local LLM using LlamaCpp
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # # Prepare the LLM

        # llm = LlamaCpp(
        #     model_path=model_path,
        #     n_gpu_layers=40,
        #     n_batch=512,
        #     n_ctx=2048,
        #     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        #     callback_manager=callback_manager,
        #     verbose=True,
        # )

        # Create chat model using OpenAI
        # llm = ChatOpenAI(temperature=0)

        # Create chat model using FireworkAI
        llm = ChatFireworks(
            model=MODEL_ID,
            model_kwargs={
                "temperature": 0,
                "max_tokens": 4096,
                "top_p": 1,
            },
        )

        return llm
