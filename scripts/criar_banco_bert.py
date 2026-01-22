import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURAÇÕES GERAIS ---
PASTA_ENTRADA = os.path.join("data", "manuais_md")      # Onde estão seus textos limpos
PASTA_SAIDA_DB = os.path.join("data", "chroma_db_bert") # Onde o banco será salvo
NOME_MODELO = "ruanchaves/bert-base-portuguese-cased-assin2-similarity"

def carregar_manuais(pasta):
    """Lê todos os arquivos .md da pasta e retorna uma lista de documentos."""
    documentos = []
    if not os.path.exists(pasta):
        print(f"❌ ERRO: Pasta '{pasta}' não encontrada.")
        return documentos
        
    arquivos = [f for f in os.listdir(pasta) if f.endswith('.md')]
    
    print(f"--- 1. Lendo arquivos da pasta '{pasta}' ---")
    for arquivo in arquivos:
        caminho = os.path.join(pasta, arquivo)
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                # Criamos um objeto 'Document' do LangChain, guardando o nome do arquivo como metadado
                doc = Document(page_content=conteudo, metadata={"fonte": arquivo})
                documentos.append(doc)
                print(f"   -> Carregado: {arquivo}")
        except Exception as e:
            print(f"   [ERRO] Falha ao ler {arquivo}: {e}")
            
    return documentos

def main():
    # 1. CARREGAR
    docs_brutos = carregar_manuais(PASTA_ENTRADA)
    if not docs_brutos:
        print("Nenhum arquivo .md encontrado! Verifique a pasta.")
        return

    # 2. DIVIDIR (CHUNKING)
    print(f"\n--- 2. Dividindo o texto em pedaços (Chunking) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Tamanho de cada pedaço
        chunk_overlap=200,    # Sobreposição
        separators=["\n\n", "\n", " ", ""] # Prioridade de corte
    )
    chunks = text_splitter.split_documents(docs_brutos)
    print(f"   -> Total de pedaços criados: {len(chunks)}")

    # 3. PREPARAR MODELO DE EMBEDDING
    print(f"\n--- 3. Inicializando Modelo BERT-PT ({NOME_MODELO}) ---")
    print("⚠️ Isso pode demorar e consumir mais memória/disco que o MiniLM...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=NOME_MODELO,
        model_kwargs={'device': 'cpu'} 
    )

    # 4. CRIAR E SALVAR O BANCO DE DADOS EM LOTES (BATCHING)
    print(f"\n--- 4. Criando e Salvando o Banco de Vetores BERT (Lotes de 50) ---")
    
    if not os.path.exists("data"):
        os.makedirs("data")

    # Inicializa o banco vazio conectado à pasta
    vector_store = Chroma(
        collection_name="apicultura_db", # Nome consistente
        embedding_function=embedding_model,
        persist_directory=PASTA_SAIDA_DB
    )

    total_chunks = len(chunks)
    tamanho_lote = 50

    # Loop para processar em fatias
    for i in range(0, total_chunks, tamanho_lote):
        lote = chunks[i : i + tamanho_lote]
        print(f"   Processando lote {i} até {min(i + tamanho_lote, total_chunks)} de {total_chunks}...")
        
        # Adiciona o lote e salva
        vector_store.add_documents(documents=lote)
        
        # Pequena pausa para o sistema respirar (bom para memória limitada)
        import time
        time.sleep(0.5)
    
    print(f"\n[SUCESSO] Banco de dados BERT criado em: {os.path.abspath(PASTA_SAIDA_DB)}")
    print("Agora você pode comparar este modelo com o MiniLM!")

if __name__ == "__main__":
    main()
