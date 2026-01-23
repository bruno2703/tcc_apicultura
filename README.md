# 🐝 Chatbot Apícola com LLM (Llama 3) e RAG

> **Trabalho de Conclusão de Curso (TCC)**
> Curso de Engenharia de Software - Universidade Federal do Ceará (Campus Quixadá)

Este repositório contém o **backend** e a inteligência artificial desenvolvidos para o projeto *"Desenvolvimento de um Chatbot com Large Language Model para Suporte Técnico Interativo a Apicultores"*.

O sistema atua como o motor de inteligência para o aplicativo móvel **Revise!**, fornecendo respostas técnicas precisas sobre manejo de abelhas no Semiárido Brasileiro, baseadas em manuais oficiais (SEBRAE, Embrapa, etc.), utilizando arquitetura de **Geração Aumentada por Recuperação (RAG)** para eliminar alucinações.

---

## 🚀 Funcionalidades Principais

*   **RAG (Retrieval-Augmented Generation):** Busca contextualmente trechos de manuais técnicos em um banco vetorial antes de gerar a resposta.
*   **Orquestração Llama 3:** Utiliza o modelo **Llama 3.3 70B** (via Groq Cloud) para processamento de linguagem natural de alto desempenho.
*   **Embeddings Especializados:** Uso de **BERT-PT** (ajustado para similaridade em Português) para uma recuperação de contexto superior a modelos multilíngues genéricos.
*   **API RESTful:** Backend em **FastAPI** pronto para integração com aplicativos móveis e web.
*   **Citação de Fontes:** Cada resposta acompanha a lista de manuais consultados, garantindo transparência ao produtor.

---

## 🏗️ Arquitetura do Sistema

O projeto é dividido em três camadas principais:

1.  **Ingestão de Dados:** Scripts que processam manuais em PDF/Markdown, realizam o chunking e salvam no ChromaDB.
2.  **Motor RAG (`rag_engine.py`):** Lógica que recebe a pergunta, recupera os 10 trechos mais relevantes e consulta o LLM.
3.  **API (`api.py`):** Interface FastAPI que expõe o serviço para o mundo exterior.

---

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python 3.9+
*   **Framework de IA:** LangChain & LangChain-Community
*   **LLM API:** Groq Cloud (Llama 3.3 70B)
*   **Banco Vetorial:** ChromaDB
*   **Modelos de Embedding:** `BERT-PT Similarity` (HuggingFace)
*   **API Framework:** FastAPI & Uvicorn

---

## 📂 Estrutura do Repositório

```text
├── app/
│   ├── api.py            # Endpoints da API FastAPI
│   └── rag_engine.py     # Lógica central do Motor RAG
├── data/
│   ├── manuals_md/       # Manuais técnicos limpos em Markdown
│   └── chroma_db_bert/   # Banco de vetores persistido (BERT)
├── scripts/
│   ├── criar_banco.py    # Gera o banco usando embeddings padrão
│   ├── criar_banco_bert.py # Gera o banco usando BERT-PT (Recomendado)
│   ├── test_rag_comparison.py # Compara RAG vs LLM Puro
│   └── test_embedding_comparison.py # Compara MiniLM vs BERT-PT
├── requirements.txt      # Dependências do projeto
└── .env                  # Chaves de API e variáveis de ambiente
```

---

## ⚙️ Configuração e Instalação

### Pré-requisitos
*   Python 3.9 ou superior.
*   Chave de API da [Groq Cloud](https://console.groq.com/).

### Passo a Passo

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/bruno2703/tcc_apicultura.git
    cd tcc_apicultura
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/scripts/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure as variáveis de ambiente:**
    Crie um arquivo `.env` na raiz do projeto com:
    ```env
    GROQ_API_KEY=sua_chave_aqui
    ```

4.  **Inicialize o Banco Vetorial (Opcional se já existir):**
    ```bash
    python scripts/criar_banco_bert.py
    ```

---

## 🚀 Como Rodar

### Iniciar o Servidor API
```bash
uvicorn app.api:app --reload
```
A API estará disponível em `http://127.0.0.1:8000`. Acesse `/docs` para a documentação interativa (Swagger).

### Endpoints Principais
*   `GET /`: Verifica status do sistema.
*   `POST /chat`: Envia pergunta e recebe resposta + fontes.
    *   *Payload:* `{"texto": "Como combater a traça da cera?"}`

---

## 📊 Resultados e Validação

Foram realizados testes rigorosos para validar a precisão do sistema:
*   **RAG vs LLM Puro:** O uso de RAG reduziu drasticamente as alucinações sobre plantas específicas do Semiárido e recomendações de dosagem de medicamentos.
*   **Embeddings BERT-PT:** Apresentou uma recuperação de contexto 30% mais precisa para termos técnicos em português comparado ao MiniLM multilíngue.

*Os logs detalhados podem ser encontrados em `rag_comparison_results.md` e `embedding_comparison_results.md`.*

---

## 🤝 Contribuição e Autoria

Este projeto foi desenvolvido por **Bruno** como parte do Trabalho de Conclusão de Curso na **UFC Quixadá**.

*   **Orientador:** Rafael Braga
*   **Instituição:** Universidade Federal do Ceará - Campus Quixadá