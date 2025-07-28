import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key = os.environ.get("OPENAI_API_KEY")
)

# Prompts mais explícitos para guiar melhor o LLM
parte1 = PromptTemplate.from_template("Resuma a seguinte queixa de cliente em uma única frase: {queixa}") | llm | StrOutputParser()
parte2 = PromptTemplate.from_template("Qual é o sentimento (Positivo, Negativo ou Neutro) da seguinte análise de queixa: {resultado_analise}") | llm | StrOutputParser()
parte3 = PromptTemplate.from_template("Formule uma resposta empática e profissional para um cliente, com base no sentimento da sua queixa, que foi: {sentimento}") | llm | StrOutputParser()


chain = (
    # Inicializo o RunnablePasstrought
    # associando toda a minha queixa que será dada no invok a este dicionário com a chave "queixa".
    {"queixa": RunnablePassthrough()} 
    | RunnablePassthrough.assign(resultado_analise=parte1)
    | RunnablePassthrough.assign(sentimento=parte2)
    | parte3
)

texto_queixa = "Hoje comprei um telefone novo, modelo X com 256 GB e flip. No entanto, o produto apresentou defeito na dobradiça e não permanece fechado. O suporte não me atende e estou super arrependido."
resultado = chain.invoke({"queixa": texto_queixa}) # O dicionário inicial é passado aqui

print(resultado)