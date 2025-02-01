import gradio as gr
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get('API_KEY')
login(token=hf_token)

pdf_paths = [
    "/content/bhagavad-gita-in-english-source-file.pdf",
    "/content/Bhagavad-gita_As_It_Is_Full.pdf"]

pages = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    current_pages = loader.load_and_split()
    pages.extend(current_pages)

for p in pages:
    p.page_content = p.page_content.replace("\n", " ")

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunked_docs = splitter.split_documents(pages)

seen_content = set()
unique_chunked_docs = []
for doc in chunked_docs:
    stripped_content = doc.page_content.strip()
    if stripped_content not in seen_content:
        seen_content.add(stripped_content)
        new_doc = Document(
            page_content=stripped_content,
            metadata=doc.metadata  
        )
        unique_chunked_docs.append(new_doc)

embed_model = HuggingFaceEmbeddings(model_name='thenlper/gte-base')
vector_store = Chroma(
    collection_name="bg_data_english",
    embedding_function=embed_model,
    persist_directory="./bg_data_english"
)
vector_store.add_documents(documents=unique_chunked_docs)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.4})    

# Load the LLM
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", cache_dir="./gemma-2-9b-it")
llm_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it", cache_dir="./gemma-2-9b-it",
    quantization_config=quantization_config,
    device_map = 'auto'
)
text_generation_pipeline = pipeline(
    model=llm_model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=False,
    max_new_tokens=500
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
Use the provided context to craft your response and remain faithful to the philosophy of the Bhagavad Gita. If you don't know the answer, humbly admit it or request the user to clarify or provide more details.
Limit your response to 5 lines unless the user explicitly asks for more explanation. Answer must be well-structured and coherent, providing a clear and concise solution to the user's query.

**Prohibited:**
- General Gita knowledge beyond provided context
- Philosophical extrapolations
- Personal interpretations

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
    colon_index = answer[:25].find(":")
    if colon_index != -1:
        answer = answer[colon_index + 1:].strip()

    chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])

    return answer

# Create Gradio interface
interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask your question", placeholder="What's troubling you?"),
    outputs=gr.Textbox(label="Answer"),
    title="Bhagavad Gita Chatbot",
    description="Get answers to your real life problems from the teachings of the Bhagavad Gita." 
)

# Launch the app
if __name__ == "__main__":
    interface.launch(debug = True, pwa = True, share = True)