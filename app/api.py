# --- Ajustes PARA O RENDER (Injetar SQLite atualizado) ---
# Só funciona no Linux (Render), no Windows usa o SQLite nativo
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass  # Windows: usa sqlite3 padrão
# ------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_engine import ApiculturaRAG

# 1. Definição dos Modelos de Dados (Entrada e Saída)
class PerguntaRequest(BaseModel):
    texto: str

class RespostaResponse(BaseModel):
    resposta: str
    fontes: list[str]

# 2. Inicialização da API
app = FastAPI(
    title="API TCC Apicultura",
    description="Backend RAG para suporte técnico a apicultores do Semiárido.",
    version="1.0.0"
)

# 3. Carregar o Motor RAG (Variável Global)
# Carregamos aqui para que ele fique na memória esperando as perguntas
motor_rag = None

@app.on_event("startup")
async def startup_event():
    """Executado quando o servidor liga."""
    global motor_rag
    print("Iniciando servidor API...")
    try:
        motor_rag = ApiculturaRAG()
    except Exception as e:
        print(f"Erro fatal ao iniciar RAG: {e}")

# 4. O Endpoint (A 'Tomada')
@app.post("/chat", response_model=RespostaResponse)
async def chat_endpoint(request: PerguntaRequest):
    """
    Recebe uma pergunta JSON e retorna a resposta do assistente.
    """
    if not motor_rag:
        raise HTTPException(status_code=500, detail="Motor RAG não foi inicializado.")
    
    # Chama a lógica do arquivo rag_engine.py
    resultado = motor_rag.gerar_resposta(request.texto)
    
    if not resultado["sucesso"]:
        raise HTTPException(status_code=500, detail=resultado.get("erro_detalhe"))
    
    return {
        "resposta": resultado["resposta"],
        "fontes": resultado["fontes"]
    }

@app.get("/")
async def root():
    return {"status": "online", "message": "API de Apicultura rodando! Acesse /docs para testar."}