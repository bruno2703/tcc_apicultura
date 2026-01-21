import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Adiciona o diretório raiz ao path para importar a app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag_engine import ApiculturaRAG

# Carrega variáveis de ambiente
load_dotenv()

def get_non_rag_response(question, llm):
    """Gera uma resposta usando o LLM diretamente sem RAG."""
    template = """Você é um Assistente Técnico Especialista em Apicultura.
    Responda a pergunta do produtor de forma didática e técnica.
    
    Pergunta: {question}
    
    Resposta Técnica:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

def main():
    print("🚀 Iniciando Teste Comparativo de RAG...")
    
    questions = [
        "Notei algumas abelhas com as asas deformadas andando na frente da colmeia. O que pode ser isso?",
        "Qual é a melhor época para fazer a colheita do mel aqui na região?",
        "Estou vendo umas formigas atacando o apiário. Posso passar qualquer veneno em volta das caixas?",
        "Minhas abelhas estão ficando sem comida por causa da seca. Como faço alimentação artificial?",
        "O enxame está muito bravo e defensivo, atacando longe da caixa. É normal?",
        "O que é puxada de cera?"
    ]
    
    # 1. Inicializa Motor RAG
    print("⏳ Inicializando Motor RAG...")
    motor_rag = ApiculturaRAG()
    
    # 2. Inicializa LLM Puro (mesmo modelo usado no RAG)
    print("🧠 Configurando LLM Puro (Llama 3 via Groq)...")
    llm_puro = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 Testando Pergunta {i}/{len(questions)}: {q}")
        
        # Teste COM RAG
        print("🔍 Consultando COM RAG...")
        res_rag = motor_rag.gerar_resposta(q)
        
        # Teste SEM RAG
        print("💬 Consultando SEM RAG...")
        res_puro = get_non_rag_response(q, llm_puro)
        
        results.append({
            "pergunta": q,
            "com_rag": res_rag['resposta'],
            "fontes": res_rag['fontes'],
            "sem_rag": res_puro
        })
    
    # 3. Salva resultados em arquivo markdown
    output_file = "rag_comparison_results.md"
    print(f"\n💾 Salvando resultados em {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Resultados do Teste: Comparação RAG vs Sem RAG\n\n")
        f.write("Este arquivo documenta a comparação entre as respostas do chatbot com o Motor RAG (usando manuais técnicos) e sem o RAG (LLM puro).\n\n")
        
        for i, res in enumerate(results, 1):
            f.write(f"## Pergunta {i}: {res['pergunta']}\n\n")
            
            f.write("### 🤖 Resposta COM RAG (Contextualizada)\n\n")
            f.write(f"{res['com_rag']}\n\n")
            f.write("> **Fontes:** " + ", ".join(res['fontes']) + "\n\n")
            
            f.write("### 🌐 Resposta SEM RAG (Llama 3 Puro)\n\n")
            f.write(f"{res['sem_rag']}\n\n")
            f.write("---\n\n")
            
    print("✅ Teste concluído com sucesso!")

if __name__ == "__main__":
    main()
