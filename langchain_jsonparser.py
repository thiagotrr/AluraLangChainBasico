# Case: agente de viagem com cadeia de LangChain 
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

set_debug(True)

# MELHORIA: Usar 'description' para descrever o campo, em vez de definir um valor padrão.
class Destino(BaseModel):
    cidade: str = Field(description="cidade a visitar")
    motivo: str = Field(description="motivo pelo qual é interessante visitar a cidade")

parser = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7
)

# 1. O prompt é criado, incluindo as instruções de formatação do parser.
prompt = PromptTemplate(
    template = """Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parser.get_format_instructions()}
)

# 2. A cadeia é montada com LCEL (|) para conectar o prompt, o LLM e o parser.
#    Este é o fluxo que garante que a saída do LLM seja processada pelo parser.
chain = prompt | llm | parser

# 3. A cadeia é invocada com um dicionário como entrada.
resultado = chain.invoke({"interesse": "praias"})

# 4. O resultado agora é um dicionário Python limpo, pronto para ser usado.
print(resultado)
print(f"\nTipo do resultado: {type(resultado)}")
print(f"Cidade: {resultado['cidade']}")
print(f"Motivo: {resultado['motivo']}")