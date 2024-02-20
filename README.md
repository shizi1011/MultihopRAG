# Chat with PDF

This is a Python project that allows you to chat with a chatbot about the PDF you uploaded using RAG (Retrieval Augmented Generation) technique. The project is built using Langchain and Streamlit framework.

## Installation

To run this project, please follow the steps below:

1. Clone the repository:

```shell
git clone https://github.com/shizi1011/Chat_With_PDF.git
cd Chat_With_PDF
```

2. Create and activate a virtual environment (optional but recommended):

```shell
python3 -m venv env
source env/bin/activate
```

3. Install the dependencies from the `requirements.txt` file:

```shell
pip install -r requirements.txt
```

4. Create `.env` file and add your API

## Running the Project

To start the application, run the following command:

```shell
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Default Configuration:

- The default Language Model (LLM) is set to `ChatFireworks` with the default model being `mistral-7b-instruct`. However, you can customize the LLM according to your needs in the `llm.py` file.
    - **vllm** : Uncomment `vllm` session in `llm.py`, then run the scripts below
    ```shell
    pip install vllm
    python -m vllm.entrypoints.api_server \
    --model MODELNAME\
    ```

    - **Llamacpp** : Uncomment `Llamacpp` session in `llm.py`, then run the scripts below
    ```shell
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python[server]
    python3 -m llama_cpp.server --model MODEL_PATH --n_gpu_layers N_GPU_LAYERS --n_ctx N_CTX --chat_format CHAT_FORMAT --host 0.0.0.0 --port 8000
    ```
    - **ChatHuggingFace** : Uncomment `ChatHuggingFace` session in `llm.py`, then run the scripts below
    Add your `HUGGINGFACE_HUB_API_TOKEN` to .env file

- The default embedding function is Nomic embedding. However, if you wish, you can use any embedding function by editing or uncommenting the appropriate sections in the `ingest.py` file.

- The default vector database is Chroma. Nevertheless, you can change the vector database as desired by editing or uncommenting the relevant sections in the `ingest.py` file.