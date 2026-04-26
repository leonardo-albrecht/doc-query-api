# Doc Query API

API de consulta inteligente a documentos via RAG (Retrieval-Augmented Generation).

## O problema que resolve
Documentos PDF são inacessíveis para busca semântica. 
Esta API ingere qualquer PDF e responde perguntas baseadas 
exclusivamente no conteúdo do documento.

## Decisões de engenharia

**Chunking com overlap**
Chunks de 500 caracteres com overlap de 50 evitam perda 
de contexto nas bordas. Sem overlap, informações que cruzam 
boundaries de chunks seriam perdidas na busca.

**Temperature 0.1**
Respostas factuais exigem baixa aleatoriedade. O modelo 
deve se ater ao contexto, não ao seu conhecimento geral.

**ChromaDB persistente**
Embeddings salvos em disco evitam re-ingestão a cada restart.

## Stack
- FastAPI
- ChromaDB
- Groq (LLaMA 3.3 70B)
- PyPDF2


## Exemplo de uso

**Ingerir um documento:**
```json
POST /ingest
{
  "file_path": "/caminho/para/documento.pdf"
}
```

**Consultar o documento:**
```json
POST /query
{
  "question": "O que é engenharia da persuasão?",
  "top_k": 3
}
```

**Resposta:**
```json
{
  "answer": "A engenharia da persuasão é...",
  "sources": ["trecho 1", "trecho 2", "trecho 3"],
  "tokens_used": 949
}
```

## Próximos passos
- [ ] Suporte a múltiplos documentos simultâneos
- [ ] Embeddings com sentence-transformers
- [ ] Deploy com Docker
- [ ] Autenticação na API

## Como rodar

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Acesse a documentação: http://localhost:8000/docs

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | /health | Status da API |
| POST | /ingest | Ingere um PDF |
| POST | /query | Consulta o documento |