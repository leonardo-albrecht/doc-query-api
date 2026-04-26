from fastapi import FastAPI, HTTPException
from app.models import IngestRequest, QueryRequest, QueryResponse
from app.ingestion import ingerir_documento
from app.retrieval import buscar_chunks
from app.generation import gerar_resposta

app = FastAPI(
    title="Doc Query API",
    description="RAG pipeline para consulta inteligente em documentos",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        resultado = ingerir_documento(request.file_path)
        return resultado
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        chunks = buscar_chunks(request.question, top_k=request.top_k)
        resultado = gerar_resposta(request.question, chunks)
        return QueryResponse(
            answer=resultado["resposta"],
            sources=chunks,
            tokens_used=resultado["tokens_usados"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))