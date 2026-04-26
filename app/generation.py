from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def gerar_resposta(pergunta: str, chunks: list[str]) -> dict:
    contexto = "\n\n".join(chunks)

    prompt = f"""Você é um assistente especializado. 
Use APENAS o contexto abaixo para responder a pergunta.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação no documento."

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # baixo → respostas mais factuais
    )

    return {
        "resposta": response.choices[0].message.content,
        "tokens_usados": response.usage.total_tokens
    }