# Case: agente de viagem com LangChain
import os
from langchain_openai import ChatOpenAI

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"
prompt = f"Crie um roteiro de viagens de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças, que gostam de {atividade}"
print(prompt)

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7,
    )

resposta = llm.invoke(prompt)
print(resposta.content)