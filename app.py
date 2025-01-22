import gradio as gr
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
import zipfile
import os

from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get('API_KEY')
login(token=hf_token)

zip_path = "/content/bg_data_english.zip"  # Path to your ZIP file
extract_path = "./bg_data_unzipped_english"  # Target folder for extraction

if not os.path.exists(extract_path):  # Check if already extracted
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

embed_model = HuggingFaceEmbeddings(model_name='thenlper/gte-base')

# Load the pre-existing vector store
vector_store = Chroma(persist_directory=extract_path, embedding_function=embed_model)
similarity_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.2}
)

# Load the LLM
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
llm_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=quantization_config,
)
text_generation_pipeline = pipeline(
    model=llm_model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=False,
    max_new_tokens=350,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Reformulating user queries with history context
rephrase_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrase_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, similarity_retriever, rephrase_prompt)

# Define the question-answering system prompt
qa_system_prompt = """You are a saintly guide inspired by the teachings of the Bhagavad Gita, offering wisdom and moral guidance. Answer questions in a friendly and compassionate tone, drawing insights from the scripture to help users with their life challenges.
Use the provided context to craft your response and remain faithful to the philosophy of the Bhagavad Gita.
If you don't know the answer, humbly admit it or request the user to clarify or provide more details.
Limit your response to 5 lines unless the user explicitly asks for more explanation.
Question:
{input}
Context:
{context}
Answer:
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
qa_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to generate answers
chat_history = []

def chat(question):
    global chat_history
    response = qa_rag_chain.invoke({"input": question, "chat_history": chat_history})
    answer = response["answer"].strip()
    if answer.startswith("Saintly Guide:"):
        answer = answer[len("Saintly Guide:"):].strip()
    elif answer.startswith("AI:"):
        answer = answer[len("AI:"):].strip()
    chat_history.extend([HumanMessage(content=question), AIMessage(content=response["answer"])])
    return answer

# Create Gradio interface
interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask your question", placeholder="What's troubling you?"),
    outputs=gr.Textbox(label="Answer"),
    title="Bhagavad Gita Chatbot",
    description="Ask questions inspired by the teachings of the Bhagavad Gita and receive saintly guidance."
)

# Launch the app
if __name__ == "__main__":
    interface.launch(debug = True)