import logging
import time
import uuid
import os
import shutil
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models import IngestRequest, QueryRequest, QueryResponse
from app.ingestion import ingerir_documento
from app.retrieval import buscar_chunks
from app.generation import gerar_resposta

# --- Structured Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Doc Query API",
    description="RAG pipeline para consulta inteligente em documentos",
    version="1.0.0",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request ID Middleware ---
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# --- Serve Frontend ---
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# --- Health ---
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# --- Upload PDF ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        resultado = ingerir_documento(file_path)

        logger.info(
            '{"event": "upload_ingest_success", "filename": "%s", "chunks": %d}',
            file.filename,
            resultado["chunks_gerados"],
        )

        return {
            "filename": file.filename,
            "chunks_gerados": resultado["chunks_gerados"],
            "caracteres_totais": resultado["caracteres_totais"],
        }
    except Exception as e:
        logger.error('{"event": "upload_error", "reason": "%s"}', str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --- Ingest (original, mantido por retrocompatibilidade) ---
@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        resultado = ingerir_documento(request.file_path)
        logger.info(
            '{"event": "ingest_success", "file_path": "%s"}',
            request.file_path,
        )
        return resultado
    except FileNotFoundError:
        logger.warning('{"event": "ingest_error", "reason": "file_not_found", "file_path": "%s"}', request.file_path)
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    except Exception as e:
        logger.error('{"event": "ingest_error", "reason": "%s"}', str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --- Query ---
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    t_total_start = time.perf_counter()

    try:
        # Etapa 1: Retrieval
        t_retrieval_start = time.perf_counter()
        chunks = buscar_chunks(request.question, top_k=request.top_k)
        retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000, 2)

        # Etapa 2: Generation
        t_generation_start = time.perf_counter()
        resultado = gerar_resposta(request.question, chunks)
        generation_ms = round((time.perf_counter() - t_generation_start) * 1000, 2)

        total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

        # Log estruturado com todas as métricas
        logger.info(
            '{"event": "query_success", "question_length": %d, "chunks_retrieved": %d, '
            '"tokens_used": %d, "retrieval_ms": %s, "generation_ms": %s, "total_ms": %s}',
            len(request.question),
            len(chunks),
            resultado["tokens_usados"],
            retrieval_ms,
            generation_ms,
            total_ms,
        )

        return QueryResponse(
            answer=resultado["resposta"],
            sources=chunks,
            tokens_used=resultado["tokens_usados"],
        )

    except Exception as e:
        total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)
        logger.error(
            '{"event": "query_error", "reason": "%s", "total_ms": %s}',
            str(e),
            total_ms,
        )
        raise HTTPException(status_code=500, detail=str(e))


# --- Mount static files AFTER routes ---
app.mount("/static", StaticFiles(directory="static"), name="static")