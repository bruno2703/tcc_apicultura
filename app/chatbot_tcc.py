import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURACOES ---
# Carrega as variaveis de ambiente do arquivo .env
load_dotenv()

# Verifica se a API key foi carregada
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY nao encontrada no arquivo .env!")

# Pega o caminho absoluto para não ter erro
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
# Sobe um nível (..) e entra em data/chroma_db
PASTA_DB = os.path.join(DIRETORIO_ATUAL, "..", "data", "chroma_db")
NOME_MODELO_EMBEDDING = "ruanchaves/bert-base-portuguese-cased-assin2-similarity"

# --- 2. CARREGAR A MEMORIA (O BANCO QUE VOCE CRIOU) ---
print("Carregando banco de dados e modelos...")

# Importante: Tem que usar o MESMO modelo de embedding que usou para criar
embedding_model = HuggingFaceEmbeddings(
    model_name=NOME_MODELO_EMBEDDING,
    model_kwargs={'device': 'cuda'}  # Mude para 'cpu' se for rodar em outro PC sem GPU
)

vector_store = Chroma(
    persist_directory=PASTA_DB,
    embedding_function=embedding_model
)

# Configura o sistema para buscar os 10 trechos mais relevantes (otimizado para melhor precisao)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# --- 3. CONFIGURAR O "CEREBRO" (LLM via GROQ) ---
# Usamos o Llama-3.3-70b pois e muito inteligente e rapido
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# --- 4. ENGENHARIA DE PROMPT (A PERSONA) ---
template = """Voce e um Assistente Tecnico Especialista em Apicultura no Semiarido Brasileiro (Caatinga).
Sua funcao e ajudar produtores rurais com base APENAS nos manuais tecnicos fornecidos.

IMPORTANTE: Voce recebera VARIOS trechos de texto abaixo. Leia TODOS com atencao antes de responder.
Mesmo que a informacao esteja fragmentada entre varios trechos, junte as pecas para dar uma resposta completa.

REGRAS DE OURO:
1. Leia TODOS os trechos do contexto antes de responder
2. Se encontrar informacoes relevantes em QUALQUER trecho, use-as
3. Combine informacoes de multiplos trechos se necessario
4. So diga "nao encontrei" se REALMENTE nenhum trecho tiver relacao com a pergunta
5. Seja didatico, pratico e incentive boas praticas de manejo
6. Cite nomes de plantas ou tecnicas especificas quando disponiveis

Contexto dos Manuais (varios trechos):
{context}

Pergunta do Produtor:
{question}

Resposta Tecnica (baseada nos trechos acima):"""

prompt = ChatPromptTemplate.from_template(template)

# --- 5. CRIAR A CHAIN (MODERNA - SEM RetrievalQA DEPRECIADO) ---
def format_docs(docs):
    """Formata documentos numerando cada trecho para facilitar leitura pelo LLM"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        fonte = doc.metadata.get('fonte', 'Desconhecida')
        formatted.append(f"[TRECHO {i} - Fonte: {fonte}]\n{doc.page_content}\n")
    return "\n".join(formatted)

# Chain RAG moderna usando LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("[OK] Sistema carregado!\n")

# --- 6. LOOP DE INTERACAO (O CHAT) ---
print("--- CHATBOT APICOLA DO SERTAO INICIADO ---")
print("Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Pergunta: ")
    if pergunta.lower() in ['sair', 'exit', 'fim']:
        print("\nEncerrando chatbot...")
        break

    print("Consultando manuais...")

    try:
        # Buscar documentos relevantes (para mostrar fontes)
        docs_relevantes = retriever.invoke(pergunta)

        # Gerar resposta usando RAG
        resposta = rag_chain.invoke(pergunta)

        print(f"\nResposta:\n{resposta}")

        # Mostrar estatisticas (bom para mostrar no TCC)
        print(f"\nTrechos consultados: {len(docs_relevantes)}")
        fontes_unicas = set(doc.metadata.get('fonte', 'Desconhecida') for doc in docs_relevantes)
        print(f"Fontes consultadas ({len(fontes_unicas)}):")
        for fonte in fontes_unicas:
            print(f"   - {fonte}")
        print("-" * 50)

    except Exception as e:
        print(f"\n[ERRO] Ocorreu um problema: {e}")
        print("Tente fazer outra pergunta.\n")
