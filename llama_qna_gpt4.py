## LLama_index class  for Question answering
# import llama_index
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


synthesis_model_id ='gpt-4'


embmodel_id='BAAI/bge-small-en-v1.5'

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def synthesis_model_fn(synthesis_model_id:str=None):
    #synthesis_model="gpt-4"
    llm = OpenAI(temperature=0.1, model=synthesis_model_id)
    return llm

def embedding_model_fn(embmodel_id:str=None):
    #embmodel_id='BAAI/bge-small-en-v1.5'
    embed_model = HuggingFaceEmbedding(model_name=embmodel_id,device=DEVICE)
    return embed_model

def node_parser_func():
    return SentenceWindowNodeParser.from_defaults(
    # how many sentences on either side to capture
    window_size=3,
    # the metadata key that holds the window of surrounding sentences
    window_metadata_key="window",
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence",
)
def prompt_helper_fnc():
    #print(type(PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)))
    return PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)


class LLama_Index_qna:
    def __init__(self,synthesis_model=None,embedding_model=None, text_splitter=None,node_parser=None,PromptHelper=None):
        self.synthesis_model =synthesis_model_fn(synthesis_model_id)
        self.embedding_model =embedding_model_fn(embmodel_id)
        print(self.embedding_model)
        self.text_splitter =SentenceSplitter(chunk_size=512, chunk_overlap=10)
        self.node_parser=node_parser_func()
        print(self.node_parser)
        self.prompt_helper =prompt_helper_fnc()
        #self.service_context = ServiceContext.from_defaults(llm=self.synthesis_model,embed_model=self.embedding_model,node_parser=self.node_parser,prompt_helper=self.prompt_helper,chunk_size=1000, chunk_overlap=200)
        self.service_context = ServiceContext.from_defaults(llm=self.synthesis_model,embed_model=self.embedding_model,node_parser=self.node_parser,prompt_helper=self.prompt_helper)
        self.persist_directory ='./storage'
        self.embd_id=embmodel_id
        print(self.service_context)

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
        try:

            shutil.rmtree(self.persist_directory)
        except:
            pass
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
            print('\n\n')
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_directory)
            index = load_index_from_storage(storage_context)
            return index

    # @classmethod
    def qna(self,query:str=None):
        print(Text.BOLD_START+Text.RED +'Entered into QNA !!!!!!!!!!!!!!' +Text.END )
        index =self.creating_index()
        print(type(index))
        print(dir(index))
        query_engine = index.as_query_engine(k=4)
        if not isinstance(query,list):
            Questions =[query]
        print('%'*99)    
        print(Questions)    
        data=[]
        for Question in Questions:
            response = query_engine.query(Question)
            final_response = {}
            final_response['Query'] =Question
            final_response['Result'] = response.response
            final_response['Filename'] = list(set([ v['file_name'] for x ,v in response.metadata.items()]))
            final_response['page_no'] = [ v['page_label'] for x ,v in response.metadata.items()]
            data.append(final_response)
        df =pd.DataFrame(data)
        embmodel_id =self.embd_id.replace('/','_')
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
    llama_obj=LLama_Index_qna()
    llama_obj.creating_index()
    llama_obj.qna(query='when did NASA  established the Planetary Defense Coordination Office?')        