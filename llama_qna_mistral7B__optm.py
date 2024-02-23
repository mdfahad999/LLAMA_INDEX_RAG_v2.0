## LLama_index class  for Question answering
# import llama_index
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index import ServiceContext, LLMPredictor,  PromptHelper,VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter,SentenceSplitter
from llama_index.readers.schema.base import Document
from llama_index import StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
import tiktoken
import os
import openai
import logging
import sys
import nltk
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.embeddings import HuggingFaceEmbedding
import shutil
from llama_index.llms import OpenAI
# from llama_index.llms.base import LLM
from llama_index.llms.utils import resolve_llm
from pydantic import BaseModel, Field
import os
from llama_index.agent import OpenAIAgent, ReActAgent
from llama_index.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    ServiceContext,
    Document,
)
from llama_index import Prompt
import torch
from typing import List, cast, Optional
from llama_index import SimpleDirectoryReader
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.types import BaseAgent
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.chat_engine import CondensePlusContextChatEngine
# from core.builder_config import BUILDER_LLM
from typing import Dict, Tuple, Any
import pandas as pd
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from faiss_vector_store import *
from llama_index import set_global_service_context
from langchain import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from chroma_vector_store import *
# path='./storage'
# try:

#     shutil.rmtree(path)
# except:
#     pass
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
class Text:
    BOLD_START = '\033[1m'
    END = '\033[0m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

from config import *
openai.api_key = OPENAI_GPT4_KEY


synthesis_model_id ='./Models/mistral-7b-instruct-v0.1.Q4_0.gguf'


# embmodel_id='thenlper/gte-large'
embmodel_id='BAAI/bge-small-en-v1.5'



class LLama_Index_qna:
    def __init__(self):
        self.embd_id = embmodel_id
        self.syn_model_id = synthesis_model_id
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.synthesis_model = self.synthesis_model_fn(llm_load='llamacpp')
        self.embedding_model = self.embedding_model_fn()
        #self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
        self.node_parser = self.node_parser_func()
        self.prompt_helper = self.prompt_helper_func()
        
        self.persist_directory = './storagess'
        self.service_context = ServiceContext.from_defaults(llm=self.synthesis_model,embed_model=self.embedding_model,node_parser=self.node_parser,prompt_helper=self.prompt_helper)
#        self.service_context = ServiceContext.from_defaults(llm=self.synthesis_model,embed_model=self.embedding_model,node_parser=self.node_parser,prompt_helper=self.prompt_helper,text_splitter=self.text_splitter)
        self.set_global_service_context = set_global_service_context(self.service_context)
    def synthesis_model_fn(self,llm_load:str=None):
            if llm_load =='llamacpp':
                llm = LlamaCPP(
                    model_url=None,
                    model_path=self.syn_model_id,
                    temperature=0.01,
                    max_new_tokens=1024,
                    context_window=8192,
                    generate_kwargs={},
                    model_kwargs={"n_gpu_layers": 5},
                    messages_to_prompt=messages_to_prompt,
                    completion_to_prompt=completion_to_prompt,
                    verbose=True
                )
                return llm
            elif llm_load== 'huggingface':
                MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
                tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=self.device,
                    quantization_config=quantization_config
                )

                generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
                generation_config.max_new_tokens = 1024
                generation_config.temperature = 0.0001
                generation_config.top_p = 0.95
                generation_config.do_sample = True
                generation_config.repetition_penalty = 1.15

                pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    return_full_text=True,
                    generation_config=generation_config,
                )
                llm = HuggingFacePipeline(
                    pipeline=pipeline,
                    )
                
                return llm    
    def embedding_model_fn(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embd_id, device=self.device)
        return embed_model 
    
    def node_parser_func(self):
        return SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
        )

    def prompt_helper_func(self):
        return PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)

    def initialize_service_context(self):
        return ServiceContext.from_defaults(
            llm=self.syn_model_id,
            embed_model=self.embd_id,
            node_parser=self.node_parser,
            prompt_helper=self.prompt_helper
        )           
    @staticmethod
    def load_data(
        file_names: Optional[List[str]] = None,
        directory: Optional[str] = None,
        urls: Optional[List[str]] = None,
    ) :
        """Load data."""
        file_names = file_names or []
        directory = directory or ""
        urls = urls or []

        # get number depending on whether specified
        num_specified = sum(1 for v in [file_names, urls, directory] if v)

        if num_specified == 0:
            raise ValueError("Must specify either file_names or urls or directory.")
        elif num_specified > 1:
            raise ValueError("Must specify only one of file_names or urls or directory.")
        elif file_names:
            reader = SimpleDirectoryReader(input_files=file_names)
            docs = reader.load_data()
        elif directory:
            reader = SimpleDirectoryReader(input_dir=directory)
            docs = reader.load_data()
        elif urls:
            from llama_hub.web.simple_web.base import SimpleWebPageReader

            # use simple web page reader from llamahub
            loader = SimpleWebPageReader()
            docs = loader.load_data(urls=urls)
        else:
            raise ValueError("Must specify either file_names or urls or directory.")

        return docs
    # @classmethod
    def creating_index(self,file_names=None):
        # path='./storage'
        # try:

        #     shutil.rmtree(self.persist_directory)
        # except:
        #     pass
        if file_names is None:
            try:
                file_names= [os.path.join(os.getcwd(),x)for x in os.listdir('.')  if x.endswith('.pdf')]
            except:
                raise ValueError(Text.BOLD_START+Text.RED +"May be file is not uplloaded!!!!"+Text.END )

        if not os.path.exists(self.persist_directory):

            print(Text.BOLD_START+Text.PURPLE +'Index is not Present .Creating it!!!!!!'+Text.END )
            # load the documents and create the index
            print(file_names)
            documents =self.load_data(file_names=file_names)
            print('Doc  :',documents)
            index = VectorStoreIndex.from_documents(
            documents,
            service_context = self.service_context,
            show_progress=True
            )
            # store it for later
            index.storage_context.persist(persist_dir=self.persist_directory)
            print('**'*123)
            return index
        else:
            print(Text.BOLD_START+Text.GREEN +'Index already exist' +Text.END )
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_directory)
            index = load_index_from_storage(storage_context)
            return index
    # @classmethod    
    def RunQuestion(self,questionText):
            
        queryQuestion = "<s>[INST] You are a AI Assistant who is specialized on answering questions from PDF Documents. Answer questions in a positive, helpful and empathetic way. Answer the following question: " + questionText.strip() + " [/INST]"
        return queryQuestion
    
    def qna_template(self):

        print('Entered!!!')
        template = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {query_str} \nContext: {context_str} \nAnswer:")
        qa_template = Prompt(template)
        return qa_template
    # @classmethod
    def qna(self,vectorstore:str,Questions:list=None):
        print(Text.BOLD_START+Text.RED +'Entered into QNA !!!!!!!!!!!!!!' +Text.END )
        if vectorstore == 'faiss':
            print(Text.BOLD_START+Text.GREEN +'Using FAISS VectorStore!!!' +Text.END )
            index=FaissVectorizer().load_from_index() 
        #docker pull chromadb/chroma    
        #docker run -p 8000:8000 chromadb/chroma     
        elif vectorstore == 'chroma':
            print(Text.BOLD_START+Text.GREEN +'Using chroma VectorStore!!!' +Text.END )
            index=ChromaDBVectorizer().load_from_index()     
        else:    
            index =self.creating_index()
        qa_template=self.qna_template()    
        ## Using Hybrid search , but hybrid search is present in weaviate and qdrant not in faiss or not sure about  chroma
        kwargs = {"similarity_top_k": 4, "vector_store_query_mode": 'hybrid','alpha':1,'qa_template':qa_template}
        query_engine = index.as_query_engine(**kwargs)
        print(Text.BOLD_START+Text.RED +'query_engine!@@@@@::' +Text.END )
        print('',query_engine)
        if not isinstance(Questions,list):
            Questions =[query1]
        #Questions=query    
        print('%'*99)    
        #print(Questions)    
        data=[]
        # Questions =["when did NASA  established the Planetary Defense Coordination Office?",'Summarize the given documents']
        for Question in Questions:
            query1=self.RunQuestion(Question)
            print(Text.BOLD_START+Text.DARKCYAN +f'{query1}' +Text.END )
            response = query_engine.query(query1)
            final_response = {}
            final_response['Query'] =Question
            final_response['Result'] = response.response
            final_response['Filename'] = list(set([ v['file_name'] for x ,v in response.metadata.items()]))
            final_response['page_no'] = [ v['page_label'] for x ,v in response.metadata.items()]
            data.append(final_response)
        df =pd.DataFrame(data)
        embmodel_id =self.embd_id.replace('/','_').replace('.','')
        synthesis_model_id=self.syn_model_id.replace('./Models/','')
        df.to_csv(f'Results/LLama_index_{embmodel_id}_{synthesis_model_id}.csv', index=False)
        #print(final_response)
        return final_response
    
# text_splitter = TokenTextSplitter(
#   separator=" ",
#   chunk_size=1024,
#   chunk_overlap=20,
#   backup_separators=["\n"],
# )



# node_parser = SimpleNodeParser.from_defaults(
#   text_splitter=text_splitter
# )


# documents = data
# index = VectorStoreIndex.from_documents(
#     documents,
#     service_context = service_context
#     )

 
# index.storage_context.persist()


# from llama_index import Prompt

# Define a custom prompt
# template = (
#     "We have provided context information below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given this information, please answer the question and each answer should start with code word AI Demos: {query_str}\n"
# )
# qa_template = Prompt(template)

# # Use the custom prompt when querying
# query_engine = custom_llm_index.as_query_engine(text_qa_template=qa_template)
        
if __name__ == '__main__':
    Questions =["when did NASA  established the Planetary Defense Coordination Office?",'Summarize the given documents']
    llama_obj=LLama_Index_qna()
    #llama_obj.creating_index()
    llama_obj.qna(vectorstore='chroma',Questions=Questions) 
    #llama_obj.qna(vectorstore='',query=Questions)        
