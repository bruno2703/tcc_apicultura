import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Carrega variáveis de ambiente
load_dotenv()

class ApiculturaRAG:
    def __init__(self):
        """
        Inicializa o motor RAG. Carrega o banco vetorial e o LLM na memória.
        Isso roda apenas uma vez quando a API inicia.
        """
        print("🚀 Inicializando Motor RAG do TCC...")
        
        # 1. Configuração de Caminhos
        self.diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        # Ajuste o caminho conforme sua estrutura: app/rag_engine.py -> data/chroma_db
        self.pasta_db = os.path.join(self.diretorio_atual, "..", "data", "chroma_db")
        
        # 2. Validar API Key
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("❌ ERRO CRÍTICO: GROQ_API_KEY não encontrada no .env!")

        # 3. Carregar Embeddings (O mesmo usado na ingestão)
        print("⏳ Carregando Embeddings (MiniLM Multilíngue)...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'} # Use 'cpu' se não tiver GPU no servidor
        )

        # 4. Conectar ao Banco de Vetores
        print(f"📂 Conectando ao ChromaDB em: {self.pasta_db}")
        self.vector_store = Chroma(
            persist_directory=self.pasta_db,
            embedding_function=self.embedding_model
        )
        
        # Recuperador otimizado (k=10 para maior precisão)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # 5. Configurar LLM (Llama 3 via Groq)
        print("🧠 Configurando Llama 3 (Groq)...")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

        # 6. Preparar a Chain (Prompt + Lógica)
        self.rag_chain = self._criar_chain()
        print("✅ Motor RAG pronto para uso!")

    def _format_docs(self, docs):
        """Formata os documentos recuperados para o contexto do prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            fonte = doc.metadata.get('fonte', 'Desconhecida')
            formatted.append(f"[TRECHO {i} - Fonte: {fonte}]\n{doc.page_content}\n")
        return "\n".join(formatted)

    def _criar_chain(self):
        """Monta o pipeline de processamento LCEL."""
        template = """Você é um Assistente Técnico Especialista em Apicultura no Semiárido Brasileiro (Caatinga).
        Sua função é ajudar produtores rurais com base APENAS nos manuais técnicos fornecidos.

        REGRAS DE OURO:
        1. Leia TODOS os trechos do contexto abaixo.
        2. Combine informações de múltiplos trechos para criar uma resposta completa.
        3. Se a resposta não estiver no contexto, diga: "Desculpe, não encontrei essa informação nos manuais."
        4. Cite nomes de plantas ou técnicas específicas quando disponíveis.
        5. Seja didático e incentive boas práticas.

        Contexto dos Manuais:
        {context}

        Pergunta do Produtor:
        {question}

        Resposta Técnica:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Chain LCEL
        chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def gerar_resposta(self, pergunta: str):
        """
        Método público que a API vai chamar.
        Recebe a pergunta e devolve a resposta + metadados.
        """
        try:
            # 1. RAG: Gera a resposta textual
            resposta_texto = self.rag_chain.invoke(pergunta)

            # 2. Fonte: Recupera os documentos usados (para citar fontes)
            docs_usados = self.retriever.invoke(pergunta)
            fontes_unicas = list(set(doc.metadata.get('fonte', 'Desconhecida') for doc in docs_usados))

            return {
                "resposta": resposta_texto,
                "fontes": fontes_unicas,
                "sucesso": True
            }
        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return {
                "resposta": "Ocorreu um erro interno no servidor ao processar sua pergunta.",
                "fontes": [],
                "sucesso": False,
                "erro_detalhe": str(e)
            }

# Pequeno bloco de teste local (só roda se você executar este arquivo diretamente)
if __name__ == "__main__":
    motor = ApiculturaRAG()
    resultado = motor.gerar_resposta("Como alimentar abelhas na seca?")
    print("\n--- Teste Local ---")
    print(resultado['resposta'])
    print("\nFontes:", resultado['fontes'])