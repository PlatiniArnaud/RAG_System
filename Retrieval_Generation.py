import subprocess

# Install Necessary packages
packages_to_install = ["openai", "python-dotenv", "pinecone-client"]
for package in packages_to_install:
    try:
        subprocess.check_call(["pip", "install", "-q", package])	
    except Exception as e:
        print(e)

# Load packages
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import textwrap


# # Mount the Drive
# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)
# # Change to project directory
# os.chdir('/content/drive/MyDrive/RAG_Project/RAG_System/')

# Load environment variables from .env file
load_dotenv('my_secrets.env')

# Access the variable
openai_key = os.getenv('OPENAI_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')
# Create OpenAI client
client = OpenAI(api_key = openai_key)
# Create Pinecone client
pinecone = Pinecone(api_key = pinecone_key)
pc_index = pinecone.Index('health-system')

# Retriever class
class RetrieveContext():

    def __init__(self, embed_model = 'text-embedding-ada-002', k = 5, vectordb_index = pc_index):
        self.embed_model = embed_model
        self.k = k
        self.index = vectordb_index

    # Function to make query embeddings
    def get_embedding(self, text, model = None):

        if model is None:
            model = self.embed_model

        text = text.replace("\n", " ")
        embedding_object =  client.embeddings.create(input = text, model=model)

        return embedding_object.data[0].embedding

    # Retrieve the context to be used for generation.
    def get_context(self, query):
        # Embed the user query
        query_embeddings = self.get_embedding(query, model = self.embed_model)
        # Retrieve similar embeddings from pinecone index
        pinecone_res = self.index.query(vector=query_embeddings, top_k = self.k, include_metadata=True)['matches']
        # Get the text chunks of the similar embeddings.
        contexts = [item['metadata']['text'] for item in pinecone_res]
        # Get the sources, document name and pages
        sources = [(item['metadata']['document_name'], item['metadata'].get('pages') or item['metadata'].get('page')) for item in pinecone_res]

        return [(contexts[i], sources[i]) for i in range(len(contexts))], query

    def augmented_query(self, user_query):
        """ Augment the query with the retrieved context. """
        context_sources,  query = self.get_context(user_query)
        formatted_contexts = []
        for context, source in context_sources:
            formatted_contexts.append(f"Context: {context}\nSource: {source}")

        augmented_prompt = "\n\n---\n\n".join(formatted_contexts) + "\n\n---\n\n" + query

        return augmented_prompt
    # Answer generation
    def generator(self, system_prompt, user_prompt, model="gpt-3.5-turbo"):

        temperature_ = 0

        completion = client.chat.completions.create(
                              model=model,
                              temperature=temperature_,
                              messages=[
                                  {
                                      "role": "system",
                                      "content": system_prompt
                                  },
                                  {
                                      "role": "user",
                                      "content": user_prompt
                                  }
                              ]
                            )
        lines = (completion.choices[0].message.content).split("\n")
        lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)

        return "\n".join("\n".join(list) for list in lists)

    # Generation with an advanced model.
    def Advanced_Generation(self, query):

        primer = f"""
          You are a slightly pedantic Scientist. A highly intelligent but flawed individal that answers
          questions based on information provided by above each question. You provide complete and consise
          answers. You make sure you give traceabe information that is why you cite your sources.
          If the answer cannot be found in the context provided, you truthfully say that
          you can't say in a polite manner without further comment or atempt to answer.
          """

        llm_model = 'gpt-4-turbo-2024-04-09'
        user_prompt = self.augmented_query(query)

        return self.generator(primer, user_prompt, llm_model).replace('\n', ' ')

query  = 'What are the malformations of the human spine?'

retriever = RetrieveContext(k=7)
