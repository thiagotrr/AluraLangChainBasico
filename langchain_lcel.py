# Case: agente de viagem com cadeia de LangChain 
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

set_debug(True)

# MELHORIA: Usar 'description' para descrever o campo, em vez de definir um valor padrão.
class Destino(BaseModel):
    cidade: str = Field(description="cidade a visitar")
    motivo: str = Field(description="motivo pelo qual é interessante visitar a cidade")

parser_cidade = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7
)

# 1. O prompt é criado, incluindo as instruções de formatação do parser.
modelo_cidade = PromptTemplate(
    template = """Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parser_cidade.get_format_instructions()}
)

modelo_restaurante = PromptTemplate(
    template = "Sugira restaurantes populares entre os locais na em {cidade}.",
    input_variables=["cidade"]
)

modelo_cultural = PromptTemplate(
    template="Sugira também atividades culturais em {cidade}.",
    input_variables=["cidade"]
)

# 2. A cadeia é montada com LCEL (|) para conectar o prompt, o LLM e o parser.
#    Este é o fluxo que garante que a saída do LLM seja processada pelo parser.
chain1 = modelo_cidade | llm | parser_cidade
chain2 = modelo_restaurante | llm | StrOutputParser()
chain3 = modelo_cultural | llm | StrOutputParser()

# Ajuste para utilizar a saída da chain 1 para as demais e faz ao mesmo tempo!
cadeia = (chain1 | { 
    "restaurantes": chain2,
    "cultural": chain3 
    })

# 3. A cadeia é invocada com um dicionário como entrada.
resultado = cadeia.invoke({"interesse": "praias"})

# 4. O resultado agora é um dicionário Python limpo, pronto para ser usado.
print(resultado)
