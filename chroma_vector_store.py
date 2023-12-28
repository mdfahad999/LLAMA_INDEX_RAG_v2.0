
from llama_index import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext,ServiceContext,PromptHelper
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from dotenv import load_dotenv, find_dotenv
# from chromadb_helper import ChromaDBHelper
from pprint import pprint
from llama_index.embeddings import HuggingFaceEmbedding
import os
import torch


import chromadb
# from llama_index import set_global_service_context

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ChromaDBHelper:
    def __init__(self):
        self.client = chromadb.HttpClient(host="localhost", port="8000")

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)

    def fetch_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)

    def create_collection(self, collection_name):
        return self.client.create_collection(name=collection_name)

class ChromaDBVectorizer:
    def __init__(self):
        helper = ChromaDBHelper()

        # else:
        try:
            self.chroma_collection = helper.fetch_collection("chroma")
        except:    
            self.chroma_collection = helper.create_collection("chroma")

        self.embmodel_id ='BAAI/bge-small-en-v1.5'
        self.filename =[os.path.join(os.getcwd(),x)for x in os.listdir('.')  if x.endswith('.pdf')]

        # define embedding function
        #embed_model = HuggingFaceEmbedding(model_name=self.embmodel_id,device=DEVICE)
        
        # load documents
        self.documents = SimpleDirectoryReader(input_files=self.filename,required_exts=['.pdf']).load_data()

        # set up ChromaVectorStore and load in data
       


    def load_from_index(self):
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # initialise service context with default values
        #self.service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=1000, chunk_overlap=200)
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context)
        return self.index  