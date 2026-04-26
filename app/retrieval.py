from app.ingestion import collection

def buscar_chunks(pergunta: str, top_k: int = 3) -> list[str]:
    """
    Busca os chunks mais relevantes para a pergunta.
    ChromaDB compara embeddings e retorna os mais próximos.
    """
    resultados = collection.query(
        query_texts=[pergunta],
        n_results=top_k
    )

    chunks = resultados["documents"][0]
    return chunks