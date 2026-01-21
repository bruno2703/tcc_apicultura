import os
import sys
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path para importar a app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag_engine import ApiculturaRAG

# Carrega variáveis de ambiente
load_dotenv()

def main():
    print("🚀 Iniciando Teste Comparativo de Modelos de Embedding...")
    
    questions = [
        "Notei algumas abelhas com as asas deformadas andando na frente da colmeia. O que pode ser isso?",
        "Qual é a melhor época para fazer a colheita do mel aqui na região?",
        "Estou vendo umas formigas atacando o apiário. Posso passar qualquer veneno em volta das caixas?",
        "Minhas abelhas estão ficando sem comida por causa da seca. Como faço alimentação artificial?",
        "O enxame está muito bravo e defensivo, atacando longe da caixa. É normal?",
        "O que é puxada de cera?"
    ]
    
    # 1. Inicializa Motor RAG - Modelo MiniLM (Atual)
    print("\n⏳ Inicializando RAG com MiniLM (Multilíngue L12)...")
    motor_minilm = ApiculturaRAG(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        pasta_db=os.path.join("data", "chroma_db")
    )
    
    # 2. Inicializa Motor RAG - Modelo BERT-PT (Recomendado pelo Professor)
    print("\n⏳ Inicializando RAG com BERT-PT (Assin2 Similarity)...")
    # Nota: Este banco deve ter sido criado previamente pelo script criar_banco_bert.py
    motor_bert = ApiculturaRAG(
        model_name="ruanchaves/bert-base-portuguese-cased-assin2-similarity",
        pasta_db=os.path.join("data", "chroma_db_bert")
    )
    
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 Testando Pergunta {i}/{len(questions)}: {q}")
        
        # Teste com MiniLM
        print("🔍 Consultando com MiniLM...")
        res_minilm = motor_minilm.gerar_resposta(q)
        
        # Teste com BERT-PT
        print("🔍 Consultando com BERT-PT...")
        res_bert = motor_bert.gerar_resposta(q)
        
        results.append({
            "pergunta": q,
            "minilm": res_minilm['resposta'],
            "fontes_minilm": res_minilm['fontes'],
            "bert": res_bert['resposta'],
            "fontes_bert": res_bert['fontes']
        })
    
    # 3. Salva resultados em arquivo markdown
    output_file = "embedding_comparison_results.md"
    print(f"\n💾 Salvando resultados em {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Comparação de Embeddings: MiniLM vs BERT-PT\n\n")
        f.write("Este teste compara o modelo de embedding atual (`MiniLM-L12-v2`) com o modelo recomendado pelo seu professor (`BERT-PT Similarity`).\n\n")
        f.write("| Característica | MiniLM (L12-v2) | BERT-PT (Assin2) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **Dimensões** | 384 | 768 |\n")
        f.write("| **Foco** | Multilíngue | Português-BR |\n\n")
        
        for i, res in enumerate(results, 1):
            f.write(f"## Pergunta {i}: {res['pergunta']}\n\n")
            
            f.write("### 🟦 Resposta com MiniLM (Atual)\n\n")
            f.write(f"{res['minilm']}\n\n")
            f.write("> **Fontes MiniLM:** " + ", ".join(res['fontes_minilm']) + "\n\n")
            
            f.write("### 🟩 Resposta com BERT-PT (Recomendado)\n\n")
            f.write(f"{res['bert']}\n\n")
            f.write("> **Fontes BERT:** " + ", ".join(res['fontes_bert']) + "\n\n")
            
            f.write("---\n\n")
            
    print("✅ Teste de comparação concluído!")

if __name__ == "__main__":
    main()
