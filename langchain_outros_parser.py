from langchain.output_parsers import DatetimeOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel
import os

# 1. Corrigido o import para o pacote `langchain_core` e o case de `DateTimeOutputParser`.
# 2. Adicionado o parâmetro `format` para especificar o formato de data desejado (DD/MM/AAAA).
#    O parser instruirá o LLM a usar este formato e o usará para converter a resposta em um objeto datetime.
parser_datetime = DatetimeOutputParser(format="%d/%m/%Y")

llm = ChatOpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.5
)

prompt = PromptTemplate(
    template="""Qual a data de fundação do clube de futebol {clube}?
    
    {formato_saida}
    """,
    input_variables=["clube"],
    partial_variables={"formato_saida": parser_datetime.get_format_instructions()}
)

# A cadeia deve terminar com o objeto parser (parser_datetime) para que a saída do LLM seja processada.
# A variável 'formato_saida' é apenas a string de instrução para o LLM.
chain = prompt | llm | parser_datetime

resultado = chain.invoke({"clube": "Liverpool FC da Inglaterra"})

# O resultado é um objeto datetime. Para exibi-lo no formato desejado, usamos strftime().
print(f"Data formatada: {resultado.strftime('%d/%m/%Y')}")
print(f"Tipo do resultado: {type(resultado)}")

class Bandeira(BaseModel):
    pais: str = Field(description="nome do país")
    cores: str = Field(description="cores da bandeira")
    historia: str = Field(description="história da bandeira")

parser_bandeira = JsonOutputParser(pydantic_object=Bandeira)

prompt = PromptTemplate(
    template="""Me ale sobre a bandeira do país {pais}.
    
    {formato_saida}
    """,
    input_variables=["pais"],
    partial_variables={"formato_saida": parser_bandeira.get_format_instructions()}
)

chain = prompt | llm | parser_bandeira
resultado = chain.invoke({"pais": "Dinamarca"})
print(f"Tipo do resultado: {type(resultado)}")
print(f"País escolhido: {resultado['pais']}")
print(f"Cores da bandeira: {resultado['cores']}")
print(f"História da bandeira: {resultado['historia']}")