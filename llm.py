
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.chat_models.fireworks import ChatFireworks

# LlamaCpp_model_path = 'PATH_TO_/models/llama-2-7b-chat.Q4_K_M.gguf'
FIREWORKS_MODEL_ID = "accounts/fireworks/models/mistral-7b-instruct-4k"
# FIREWORKS_MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


class Loadllm:
    @staticmethod
    def load_llm():

        # Create Local LLM using LlamaCpp, uncomment code below to use
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # llm = LlamaCpp(
        #     model_path=model_path,
        #     n_gpu_layers=40,
        #     n_batch=512,
        #     n_ctx=2048,
        #     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        #     callback_manager=callback_manager,
        #     verbose=True,
        # )

        # Create Local LLM using vllm, uncomment code below to use 
        # inference_server_url = "http://localhost:8000/v1"
        # openai_api_key = "EMPTY"
        # model_name = "HuggingFaceH4/zephyr-7b-beta"
        # llm = ChatOpenAI(
        #     model=model_name,
        #     openai_api_key=openai_api_key,
        #     openai_api_base=inference_server_url,
        #     max_tokens=8192,
        #     temperature=0,
        # )

        # Create chat model using OpenAI, uncomment code below to use
        # llm = ChatOpenAI(temperature=0)

        # Create chat model using FireworkAI, uncomment code below to use
        llm = ChatFireworks(
            model=FIREWORKS_MODEL_ID,
            model_kwargs={
                "temperature": 0,
                "max_tokens": 4096,
                "top_p": 1,
            },
        )

        return llm
