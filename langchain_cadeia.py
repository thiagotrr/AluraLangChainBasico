# Case: agente de viagem com cadeia de LangChain 
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug

llm = ChatOpenAI(
    api_key=os.environ.get("OPEN_API_KEY"),
    model="gpt-4o",
    temperature=0.7
)
set_debug(True)

# Cada um desses é um passo separado, mas que deverão ser conectados.
modelo_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interesse}"
)

modelo_restaurante = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre os locais na em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades culturais em {cidade}"
)

cadeia_cidade      = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurante = LLMChain(prompt=modelo_restaurante, llm=llm)
cadeia_cultural    = LLMChain(prompt=modelo_cultural, llm=llm)

# verbose=True mostra os detalhes da execução para fins de debug e entendimento.
cadeia = SimpleSequentialChain(chains=[cadeia_cidade, cadeia_restaurante, cadeia_cultural], verbose=True)

# Cada cadeira poderá utilizar uma LLM distinta
resultado = cadeia.invoke("praias")
print(resultado.content)