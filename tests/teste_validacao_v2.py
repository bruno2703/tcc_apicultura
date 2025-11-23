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

PASTA_DB = "chroma_db"
NOME_MODELO_EMBEDDING = "ruanchaves/bert-base-portuguese-cased-assin2-similarity"

print("="*80)
print("TESTE DE VALIDACAO TECNICA - CHATBOT RAG")
print("="*80)
print("\nCarregando banco de dados e modelos...")

# --- 2. CARREGAR A MEMORIA ---
embedding_model = HuggingFaceEmbeddings(
    model_name=NOME_MODELO_EMBEDDING,
    model_kwargs={'device': 'cuda'}
)

vector_store = Chroma(
    persist_directory=PASTA_DB,
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 3. CONFIGURAR O LLM ---
# Usando llama-3.3-70b-versatile (modelo mais recente e rapido)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# --- 4. PROMPT TEMPLATE ---
template = """Voce e um Assistente Tecnico Especialista em Apicultura no Semiarido Brasileiro (Caatinga).
Sua funcao e ajudar produtores rurais com base APENAS nos manuais tecnicos fornecidos.

REGRAS DE OURO:
1. Use as informacoes do contexto abaixo para responder.
2. Se a resposta nao estiver no contexto, diga: "Desculpe, nao encontrei essa informacao nos manuais de referencia."
3. Seja didatico, pratico e incentive boas praticas de manejo.
4. Cite nomes de plantas ou tecnicas especificas se estiverem no texto.

Contexto dos Manuais:
{context}

Pergunta do Produtor:
{question}

Resposta Tecnica:"""

prompt = ChatPromptTemplate.from_template(template)

# --- 5. CRIAR CHAIN ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("[OK] Sistema carregado!\n")

# --- 6. TESTES DE VALIDACAO TECNICA ---
perguntas_teste = [
    "Quais plantas florescem em Janeiro na Caatinga?",
    "Como devo alimentar as abelhas na epoca da seca?",
    "O que e a planta Jurema Preta?"
]

print("="*80)
print("INICIANDO TESTES DE VALIDACAO")
print("="*80)

for i, pergunta in enumerate(perguntas_teste, 1):
    print(f"\n{'='*80}")
    print(f"TESTE {i}/3")
    print(f"{'='*80}")
    print(f"\nPergunta: {pergunta}")
    print("\nConsultando manuais...")

    try:
        # Buscar documentos relevantes
        docs = retriever.invoke(pergunta)

        # Gerar resposta
        resposta = rag_chain.invoke(pergunta)

        print(f"\nResposta:")
        print("-"*80)
        print(resposta)
        print("-"*80)

        print("\nFontes Consultadas:")
        for doc in docs:
            print(f"   - {doc.metadata.get('fonte', 'Desconhecida')}")

    except Exception as e:
        print(f"\n[ERRO] Falha na consulta: {e}")

    print("\n")

print("\n" + "="*80)
print("TESTES CONCLUIDOS!")
print("="*80)
