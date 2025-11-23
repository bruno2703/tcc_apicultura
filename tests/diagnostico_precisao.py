import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Carrega as variaveis de ambiente do arquivo .env
load_dotenv()

# Verifica se a API key foi carregada
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY nao encontrada no arquivo .env!")

print("="*80)
print("DIAGNOSTICO DE PRECISAO DO RAG")
print("="*80)

# Carregar banco
embedding_model = HuggingFaceEmbeddings(
    model_name="ruanchaves/bert-base-portuguese-cased-assin2-similarity",
    model_kwargs={'device': 'cuda'}
)

vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

print(f"\nTotal de chunks no banco: {vector_store._collection.count()}")

# Perguntas de teste que DEVEM ter resposta nos manuais
perguntas_teste = [
    "Como alimentar as abelhas na seca?",
    "Qual a receita do xarope de acucar?",
    "Quais plantas da Caatinga tem florada?",
    "Como fazer captura de enxames?",
    "O que e Juazeiro?"
]

print("\n" + "="*80)
print("TESTANDO QUALIDADE DA BUSCA VETORIAL")
print("="*80)

for i, pergunta in enumerate(perguntas_teste, 1):
    print(f"\n{'='*80}")
    print(f"TESTE {i}: {pergunta}")
    print(f"{'='*80}")

    # Testar com diferentes valores de k
    for k in [3, 5, 10]:
        print(f"\n--- Buscando top {k} documentos ---")
        docs = vector_store.similarity_search_with_score(pergunta, k=k)

        print(f"Resultado para k={k}:")
        for j, (doc, score) in enumerate(docs[:3], 1):  # Mostrar apenas os 3 primeiros
            print(f"  [{j}] Score: {score:.4f} | Fonte: {doc.metadata.get('fonte', 'N/A')}")
            # Mostrar trecho do conteudo
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"      Preview: {preview}...")

    print()

print("\n" + "="*80)
print("ANALISE:")
print("="*80)
print("""
Se os SCORES estao MUITO ALTOS (> 1.0): Busca esta ruim
Se os SCORES estao BAIXOS (< 0.5): Busca esta boa
Se os documentos retornados NAO tem relacao com a pergunta: Problema nos embeddings

RECOMENDACOES:
1. Se scores ruins: Aumentar k (numero de documentos recuperados)
2. Se textos irrelevantes: Melhorar chunking (tamanho/overlap)
3. Se modelo nao entende contexto: Trocar modelo de embedding
""")
