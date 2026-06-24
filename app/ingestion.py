import PyPDF2
from groq import Groq
import chromadb
from dotenv import load_dotenv
import os
import re


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("documentos")

def limpar_texto(texto: str) -> str:
    # Remove quebras de linha excessivas
    texto = re.sub(r'\n+', ' ', texto)
    # Remove espaços múltiplos
    texto = re.sub(r' +', ' ', texto)
    # Remove espaços antes de pontuação
    texto = re.sub(r' ([.,;:!?])', r'\1', texto)
    return texto.strip()

def extrair_texto_pdf(file_path: str) -> str:
    texto = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            texto += page.extract_text() or ""
    return texto

def chunk_texto(texto: str, tamanho: int = 500, overlap: int = 50) -> list[str]:
    """
    Divide o texto em pedaços com overlap.
    Overlap evita perder contexto nas bordas dos chunks.
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunk = texto[inicio:fim]
        chunks.append(chunk)
        inicio += tamanho - overlap  # volta 'overlap' caracteres
    return chunks

def gerar_embedding(texto: str) -> list[float]:
    """
    Groq não tem API de embeddings — usamos chromadb com embedding local.
    Em produção: trocar por OpenAI embeddings ou sentence-transformers.
    """
    return None  # ChromaDB vai gerar embeddings automaticamente

def ingerir_documento(file_path: str) -> dict:
    texto = extrair_texto_pdf(file_path)
    texto = limpar_texto(texto)   
    chunks = chunk_texto(texto)

    # Use unique prefix to prevent conflict across different document uploads
    doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.basename(file_path))
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    
    # Process and add chunks in small batches to avoid ONNX Runtime memory issues
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )

    return {
        "chunks_gerados": len(chunks),
        "caracteres_totais": len(texto)
    }