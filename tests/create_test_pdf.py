"""
Generates a minimal test PDF about 'Engenharia da Persuasão' for the eval suite.
Uses only standard library + pypdf2 (already installed).
"""
import os

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


CONTENT = """Engenharia da Persuasão

A engenharia da persuasão é o uso sistemático de técnicas psicológicas para influenciar decisões e comportamentos humanos. Ela combina princípios de psicologia cognitiva, neurociência e comunicação estratégica para criar mensagens que movem pessoas à ação.

Como o Cérebro Humano Toma Decisões de Compra

O cérebro humano combina processos emocionais e lógicos para avaliar riscos e recompensas antes de tomar uma decisão de compra. O sistema límbico processa emoções e desejos, enquanto o córtex pré-frontal aplica análise racional. Na prática, a maioria das decisões de compra começa com uma resposta emocional que é depois justificada pela lógica.

O Efeito GAP de Danny Iny

O efeito GAP, criado por Danny Iny, descreve a distância entre a situação atual do leitor e a situação desejada. Essa técnica de engajamento e contemplação funciona criando uma tensão psicológica que motiva o leitor a buscar a solução oferecida. O GAP explora três elementos: a dor atual, a promessa futura, e o caminho entre os dois.

Princípios Fundamentais

1. Reciprocidade: Quando alguém recebe algo, sente obrigação de retribuir.
2. Escassez: Recursos limitados são percebidos como mais valiosos.
3. Autoridade: Pessoas confiam em especialistas e figuras de autoridade.
4. Compromisso: Pequenos compromissos levam a compromissos maiores.
5. Prova Social: Pessoas seguem o comportamento da maioria.
6. Afinidade: Compramos de quem gostamos e confiamos.
"""


def create_test_pdf(output_path: str) -> None:
    """Create a simple text file as PDF substitute, or a real PDF if reportlab is available."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if HAS_REPORTLAB:
        c = rl_canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        y = height - 72
        for line in CONTENT.strip().split("\n"):
            if y < 72:
                c.showPage()
                y = height - 72
            c.setFont("Helvetica", 11)
            c.drawString(72, y, line.strip())
            y -= 16
        c.save()
        print(f"PDF (reportlab) created: {output_path}")
    else:
        # Fallback: create a minimal PDF manually
        # This is a bare-bones valid PDF with the text content
        lines = CONTENT.strip().replace("\n\n", "\n").split("\n")
        text_stream = "\n".join(f"({line.strip()}) Tj T*" for line in lines if line.strip())

        pdf = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length {len(text_stream) + 50} >>
stream
BT
/F1 10 Tf
72 720 Td
12 TL
{text_stream}
ET
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
{400 + len(text_stream)}
%%EOF"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pdf)
        print(f"PDF (minimal) created: {output_path}")


if __name__ == "__main__":
    create_test_pdf("tests/fixtures/persuasao.pdf")
    print("Done!")
