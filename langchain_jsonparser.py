# Case: agente de viagem com cadeia de LangChain 
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

#set_debug(True)

class Destino(BaseModel):
    cidade: str = Field("cidade a visitar", )
    motivo: str = Field("motivo pelo qual é interessante visitar a cidade")

meu_parser = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(
    api_key=os.environ.get("OPEN_API_KEY"),
    model="gpt-4o",
    temperature=0.7
)

# Cada um desses é um passo separado, mas que deverão ser conectados.
modelo_cidade = PromptTemplate(
    template = """Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": meu_parser.get_format_instructions()},
    parser=meu_parser
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
# neste exemplo, deixamos somente a cadeia da cidade.
cadeia = SimpleSequentialChain(chains=[cadeia_cidade])

# Cada cadeira poderá utilizar uma LLM distinta
resultado = cadeia.invoke("praias")
print(resultado)