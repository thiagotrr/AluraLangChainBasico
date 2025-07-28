# Case: agente de viagem
import os
from openai import OpenAI

#Exemplo básico de uma chamda direta.
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"
prompt = f"Crie um roteiro de viagens de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças, que gostam de {atividade}"
print(prompt)

cliente = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
resposta = cliente.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        # Instrução para o LLM
        { "role": "system", "content": "Você é um especialista em roteiros de viagens"},

        # Preparação para receber o prompt
        { "role": "user", "content": prompt, }
    ]
)

roteiro_viagem = resposta.choices[0].message.content
print(roteiro_viagem)