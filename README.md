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

**Inserção em lotes (Batch Ingestion)**
Para evitar falhas de alocação de memória no ONNX Runtime (`ONNXRuntimeError`) ao processar documentos extensos de uma só vez, a ingestão divide os chunks em lotes controlados de 20 unidades antes de adicionar ao ChromaDB.

**Prefixos únicos para chunks**
Geração de IDs baseada em hash/nome limpo do arquivo que previne colisões entre múltiplos uploads de documentos.

**Temperature 0.1**
Respostas factuais exigem baixa aleatoriedade. O modelo 
deve se ater ao contexto, não ao seu conhecimento geral.

**ChromaDB persistente**
Embeddings salvos em disco evitam re-ingestão a cada restart.

## Stack
- **Backend:** FastAPI, ChromaDB, PyPDF2
- **IA/LLM:** Groq (LLaMA 3.3 70B)
- **Frontend:** HTML5, CSS3 (Premium Dark-mode, Glassmorphism, CSS Transitions), Vanilla JS (Drag-and-drop, UI animada, cache local de histórico)

## Frontend Web Premium

O projeto inclui uma interface web moderna integrada diretamente à API:

- **Upload interativo:** Arraste ou selecione qualquer arquivo PDF. A interface renderiza uma barra de progresso com animação em tempo real.
- **Chat em tempo real:** Envie perguntas ao PDF de forma limpa, com indicadores de digitação (typing indicator).
- **Métricas e Fontes:** Veja a latência da consulta, a contagem exata de tokens utilizados e expanda o painel lateral para ler os chunks originais do PDF que fundamentaram a resposta da IA.
- **Histórico Persistente:** O painel lateral armazena o histórico das suas perguntas localmente no navegador, facilitando a re-consulta.

---

## Como rodar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Configure o seu arquivo `.env` com a API Key da Groq:
```env
GROQ_API_KEY=gsk_...
```

3. Inicie o servidor de desenvolvimento:
```bash
uvicorn app.main:app --reload
```

4. Acesse no navegador:
- **Interface Web:** [http://localhost:8000](http://localhost:8000)
- **Documentação interativa da API:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | / | Serve a Interface Web |
| GET | /health | Status da API |
| POST | /upload | Upload interativo de PDF (via multipart/form-data) |
| POST | /ingest | Ingestão programática via caminho de arquivo |
| POST | /query | Consulta semântica sobre o documento ativo |

## Avaliação RAG — LLM-as-a-Judge

O pipeline inclui uma suite de avaliação automatizada que usa **LLM-as-a-Judge** 
para medir a qualidade das respostas geradas pelo RAG.

### Métricas avaliadas

| Métrica | Descrição | Escala |
|---------|-----------|--------|
| **Faithfulness** | A resposta usa APENAS informações do contexto recuperado? | 1-5 |
| **Answer Relevance** | A resposta endereça diretamente a pergunta do usuário? | 1-5 |

O juiz é o próprio LLaMA 3.3 70B via Groq, com `response_format: json_object` 
para garantir saída estruturada e parsável.

### Como rodar a avaliação

```bash
python tests/evals.py
```

### Saídas geradas

- `tests/results/eval_<timestamp>.json` — dados detalhados de cada caso
- `tests/results/report.html` — dashboard visual interativo (dark mode)

## Próximos passos
- [x] Interface Web Premium (Dark mode + upload funcional)
- [x] Ingestão resiliente em lotes (Prevenção de ONNX Memory Errors)
- [ ] Suporte a múltiplos documentos simultâneos
- [ ] Deploy com Docker
- [ ] Autenticação na API

## Docker

```bash
docker build -t doc-query-api .
docker run -p 8000:8000 --env-file .env doc-query-api
```