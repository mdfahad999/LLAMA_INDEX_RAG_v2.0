import faiss
from llama_index import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext,ServiceContext,PromptHelper
from llama_index.vector_stores.faiss import FaissVectorStore
from pprint import pprint
import os
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.embeddings import HuggingFaceEmbedding
import torch
from llama_index import set_global_service_context

DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embmodel_id='BAAI/bge-small-en-v1.5'
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
####FaissVectorizer

class FaissVectorizer:
    def __init__(self):
        # dimensions of text-ada-embedding-002
        dim = 384
        faiss_index = faiss.IndexFlatL2(dim)
        self.persist_directory ="./faiss"
        self.filename =[os.path.join(os.getcwd(),x)for x in os.listdir('.')  if x.endswith('.pdf')]
        # load documents
        documents = SimpleDirectoryReader(input_files=self.filename,required_exts=['.pdf']).load_data()
        self.embedding_model =embedding_model_fn(embmodel_id)
        #self.text_splitter =SentenceSplitter(chunk_size=512, chunk_overlap=10)
        self.node_parser=node_parser_func()
        self.prompt_helper =prompt_helper_fnc()
        # create storage_context and faiss_index
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.service_context = ServiceContext.from_defaults(embed_model=self.embedding_model,node_parser=self.node_parser,prompt_helper=self.prompt_helper)
        self.index = VectorStoreIndex.from_documents(   documents, storage_context=self.storage_context)
        # save index to disk
        self.index.storage_context.persist(persist_dir=self.persist_directory)
        self.set_global_service_context=set_global_service_context(self.service_context)

    def load_from_index(self):
        # load index from disk
        self.vector_store = FaissVectorStore.from_persist_path(persist_path=self.persist_directory+'/default__vector_store.json')
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store, persist_dir=self.persist_directory)
        self.index = load_index_from_storage(storage_context=self.storage_context)
        return self.index    