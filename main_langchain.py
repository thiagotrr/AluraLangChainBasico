# Case: agente de viagem com LangChain
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

modelo_prompt = PromptTemplate.from_template(
    "Crie um roteiro de viagens de {dias} dias, para uma família com {criancas} crianças, que gostam de {atividade}"
)

prompt = modelo_prompt.format(
    dias=numero_de_dias,
    criancas=numero_de_criancas,
    atividade=atividade,
)

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), #Registrado nas variáveis de ambiente do computador.
    model="gpt-4o",
    temperature=0.7,
    )

resposta = llm.invoke(prompt)
print(resposta.content)