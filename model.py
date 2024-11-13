from langchain_openai import ChatOpenAI
# from langchain.llms import HuggingFaceHub, LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
# from langchain_core.runnables import RunnablePassthrough


from utils import Parser, deduplicate, format_docs
from promp import query_template, answer_decision_template, answer_template
# from langchain_community.llms import OpenAI

class LLMLoader:
    def __init__(self):
        pass        

    @staticmethod
    def from_openai(openai_api_key, model_name="gpt-3.5-turbo"):
        """
        Load an OpenAI model using LangChain.
        
        :param openai_api_key: OpenAI API key.
        :param model_name: OpenAI model name (default is "gpt-3.5-turbo").
        """
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=model_name,
            temperature=0.7,
            # streaming=True
        )
        return llm

    @staticmethod
    def from_huggingface_llm(model_name, huggingface_api_token):
        """
        Load a HuggingFace model using LangChain's HuggingFaceHub integration.
        
        :param model_name: Name of the model on HuggingFace Hub.
        :param huggingface_api_token: API token for HuggingFace Hub.
        """
        # llm = HuggingFaceHub(
        #     repo_id=model_name,
        #     huggingfacehub_api_token=huggingface_api_token,
        #     model_kwargs={"temperature": 0.7, "max_new_tokens": 200}
        # )
        # return llm
        pass

    @staticmethod
    def from_llamacpp(model_path, n_ctx=512):
        """
        Load a LLaMA model using LangChain's LlamaCpp integration.
        
        :param model_path: Path to the LLaMA model file.
        :param n_ctx: Context size for the LLaMA model (default is 512).
        """
        # llm = LlamaCpp(
        #     model_path=model_path,
        #     n_ctx=n_ctx,
        #     temperature=0.7
        # )
        # return llm
        pass

class EmbeddingLoader:
    def __init__(self):
        pass
    @staticmethod
    def from_huggingface_embedding(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Load HuggingFace embeddings using LangChain's HuggingFaceEmbeddings integration.

        :param model_name: The name or path of the HuggingFace model for generating embeddings.
        :param device: The device to run the model on ('cpu' or 'cuda').
        :return: An instance of HuggingFaceEmbeddings.
        """
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
        return embeddings

    @staticmethod
    def from_huggingface_bge_embedding(model_name: str, device: str = "cpu"):
        """
        Load HuggingFace embeddings using LangChain's HuggingFaceEmbeddings integration.
        :param model_name: The name or path of the HuggingFace model for generating embeddings.
        :param device: The device to run the model on ('cpu' or 'cuda').
        :return: An instance of HuggingFaceEmbeddings.
        """
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={"device": device})
        return embeddings
        
    

        
# v1
class MultihopRAGSystem:
    def __init__(self, max_hops=2, passages_per_hop=3,llm = None, retriever=None):
        self.max_hops = max_hops
        self.passages_per_hop = passages_per_hop
        self.retriever = retriever  # Assuming we have a retriever for fetching context

        # Initialize language model (LLM) and chains
        self.llm = llm  # You can replace with any model like GPT-3.5

        self.query_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(query_template),
        ])
        self.answer_decision_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(answer_decision_template),
        ])
        self.answer_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(answer_template),
        ])

        # query_chain_from_docs = (
        #     {
        #         "input": lambda x: x["input"],  # input query
        #         "context": lambda x: format_docs(x["context"]),  # context
        #     }
        #     | self.query_prompt # format query and context into prompt
        #     | self.llm # generate response
        #     | Parser()  # coerce to string
        # )
        # retrieve_docs = (lambda x: x["input"]) | self.retriever

        # self.query_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        #     answer=query_chain_from_docs
        # )


        # Initialize LLMChains for each step
        self.query_chain = self.query_prompt | self.llm | Parser()

        self.answer_decision_chain = self.answer_decision_prompt | self.llm | Parser()

        self.answer_chain = self.answer_prompt |self.llm| Parser()

    # def forward(self, question):
    #     context = []
    #     hop = 1
    #     while hop <= self.max_hops:
    #         # Generate a query based on the current context and question
    #         if len(context) == 0:
    #           query = self.query_chain.invoke({"context": 'N/A', "question": question})
    #         else:
    #           query = self.query_chain.invoke({"context": '\n'.join(context), "question": question})


    #         # Retrieve passages using the query
    #         passages = self.retriever(query).passages
    #         context = deduplicate(context+passages)

    #         # Decide whether the current context is sufficient to answer the question
    #         answer_decision = self.answer_decision_chain.invoke({"context": '\n'.join(context), "question": question})
    #         if answer_decision == "Yes":
    #             break

    #         hop += 1  # Continue to the next hop if context is insufficient

    #     # Generate the final answer from the gathered context
    #     answer = self.answer_chain.invoke({"context": context, "question": question})

    #     return answer, context
    # async def forward(self, question):
    #     context = []
    #     hop = 1
    #     while hop <= self.max_hops:
    #         # Generate a query based on the current context and question
    #         if len(context) == 0:
    #             query = await self.query_chain.ainvoke({"context": 'N/A', "question": question})
    #         else:
    #             query = await self.query_chain.ainvoke({"context": '\n'.join(context), "question": question})
    #         print(query)
    #         # Retrieve passages using the query
    #         passages = (await self.retriever.ainvoke(query))
    #         passages = [p.page_content for p in passages]
    #         context = deduplicate(context + passages)

    #         # Decide whether the current context is sufficient to answer the question
    #         answer_decision = await self.answer_decision_chain.ainvoke({"context": '\n'.join(context), "question": question})
    #         if answer_decision == "Yes":
    #             break

    #         hop += 1  # Continue to the next hop if context is insufficient

    #     # Generate the final answer from the gathered context
    #     answer = await self.answer_chain.ainvoke({"context": context, "question": question})

    #     return answer, context

    async def forward(self, question,callbacks=None):
        context = []
        hop = 1
        while hop <= self.max_hops:
            # Generate a query based on the current context and question
            if len(context) == 0:
                query = await self.query_chain.ainvoke({"context": 'N/A', "question": question},callbacks=callbacks)
            else:
                query = await self.query_chain.ainvoke({"context": '\n'.join(context), "question": question},callbacks=callbacks)
            print(query)
            # Retrieve passages using the query
            passages = (await self.retriever.ainvoke(query))
            passages = [p.page_content for p in passages]
            context = deduplicate(context + passages)

            # Decide whether the current context is sufficient to answer the question
            answer_decision = await self.answer_decision_chain.ainvoke({"context": '\n'.join(context), "question": question},callbacks=callbacks)
            if answer_decision == "Yes":
                break

            hop += 1  # Continue to the next hop if context is insufficient

        # Generate the final answer from the gathered context
        answer = await self.answer_chain.ainvoke({"context": context, "question": question},callbacks=callbacks)

        return answer, context
        
