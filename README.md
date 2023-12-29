# LLama_Index_RAG
Implementing RAG (Retrieval-Augmented Generation) through LLama_index, harnessing the capabilities of GPT-4 and Mistral/Open-source models.


## 💻 Setup

1.Create a directory named "Models" and download the Mistral 7B Open Source model, available in both 4-bit and 8-bit quantized versions with GGUF, from the following link: [Mistral 7B Quantized GGUF Model]:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf.

2.Create Conda Environment
```
conda create --name Llama-index-RAG python==3.10
pip install -r requirements.txt 
```

# Run Chroma through Docker:
1.Pull ChromaDB Docker Image
```
docker pull chromadb/chroma    
```
2.Run ChromaDB Docker
```
docker run -p 8000:8000 chromadb/chroma  
```



## 💻 Run Mistral 7B Model
```
python3  llama_qna_mixtral7B_faiss_optm.py
```
Feel free to follow these instructions to set up and run LLama_Index_RAG efficiently.