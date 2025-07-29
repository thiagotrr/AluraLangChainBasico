# Case: agente de viagem com cadeia de LangChain 
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain #Novo caminho de import.
from langchain.memory.summary import ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

#set_debug(True)

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7
)

# Sequência de mensagens
mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

# Padrão 5
memory = ConversationSummaryMemory(k=2)

conversation = ConversationChain(llm = llm, 
                                 verbose=True,
                                 memory = memory)

longa_conversa = ""
for mensagem in mensagens:
    resposta = conversation.predict(input = mensagem)
    print(resposta)

print(memory.load_memory_variables({}))