import os
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
        chunk_size=1000,      # Tamanho de cada pedaço (aprox. 2 ou 3 parágrafos)
        chunk_overlap=200,    # Sobreposição para não perder contexto
        separators=["\n\n", "\n", " ", ""] # Prioridade de corte
    )
    chunks = text_splitter.split_documents(docs_brutos)
    print(f"   -> Total de pedaços criados: {len(chunks)}")

    # 3. PREPARAR MODELO DE EMBEDDING
    print(f"\n--- 3. Inicializando Modelo de IA ({NOME_MODELO}) ---")
    embedding_model = HuggingFaceEmbeddings(
        model_name=NOME_MODELO,
        model_kwargs={'device': 'cpu'} # Use 'cpu' para compatibilidade Render
    )

    # 4. CRIAR E SALVAR O BANCO DE DADOS (CHROMA)
    print(f"\n--- 4. Criando e Salvando o Banco de Vetores (Isso pode demorar) ---")
    
    # Esta linha faz tudo: Vetoriza e Salva no disco na pasta 'chroma_db'
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PASTA_SAIDA_DB
    )
    
    print(f"\n[SUCESSO] Banco de dados criado e salvo em: {os.path.abspath(PASTA_SAIDA_DB)}")
    print("Agora você pode usar este banco para fazer perguntas!")

if __name__ == "__main__":
    main()