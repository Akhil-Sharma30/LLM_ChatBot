
from llama_index import VectorStoreIndex,download_loader, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from pathlib import Path
from github import Github
import os
import shutil
import openai
import gradio as gr

from pathlib import Path
from llama_index import download_loader

"""# Github Configeration"""

openai.api_key = os.environ.get("OPENAPI_API_KEY")

# username = 'Akhil-Sharma30'


"""# Reading the Files for LLM Model"""


# Specify the path to the repository
repo_dir = "/content/Akhil-Sharma30.github.io"

# Check if the repository exists and delete it if it does
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)


# def combine_md_files(folder_path):
#     MarkdownReader = download_loader("MarkdownReader")
#     loader = MarkdownReader()

#     md_files = [file for file in folder_path.glob('*.md')]
#     documents = None

#     for file_path in md_files:
#         document = loader.load_data(file=file_path)
#         documents += document

#     return documents

# folder_path = Path('/content/Akhil-Sharma30.github.io/content')
#combined_documents = combine_md_files(folder_path)

# combined_documents will be a list containing the contents of all .md files in the folder

MarkdownReader = download_loader("MarkdownReader")

loader = MarkdownReader()
document1 = loader.load_data(file=Path('Akhil-Sharma30.github.io/assets/README.md'))
document2 = loader.load_data(file=Path('Akhil-Sharma30.github.io/content/about.md'))
document3 = loader.load_data(file=Path('Akhil-Sharma30.github.io/content/cv.md'))
document4 = loader.load_data(file=Path('Akhil-Sharma30.github.io/content/post.md'))
document5 = loader.load_data(file=Path('Akhil-Sharma30.github.io/content/opensource.md'))
document6 = loader.load_data(file=Path('Akhil-Sharma30.github.io/content/supervised.md'))

data = document1+ document2 + document3+ document4 + document5+document6


"""# Vector Embedding"""

index = VectorStoreIndex.from_documents(data)

query_engine = index.as_query_engine()
response = query_engine.query("know akhil?")
print(response)

response = query_engine.query("what is name of the person?")
print(response)

"""# ChatBot Interface"""

def chat(chat_history, user_input):

  bot_response = query_engine.query(user_input)
  #print(bot_response)
  response = ""
  for letter in ''.join(bot_response.response): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
      response += letter + ""
      yield chat_history + [(user_input, response)]

with gr.Blocks() as demo:
    gr.Markdown('# Robotic Akhil')
    gr.Markdown('## "Innovating Intelligence - Unveil the secrets of a cutting-edge ChatBot project that introduces you to the genius behind the machine. 👨🏻‍💻😎')
    gr.Markdown('> Hint: Akhil 2.0')
    gr.Markdown('## Some question you can ask to test Bot:')
    gr.Markdown('#### :) know akhil?')
    gr.Markdown('#### :) write about my work at Agnisys?')
    gr.Markdown('#### :) write about my work at IIT Delhi?')
    gr.Markdown('#### :) was work in P1 Virtual Civilization Initiative opensource?')
    gr.Markdown('#### many more......')
    with gr.Tab("Knowledge Bot"):
#inputbox = gr.Textbox("Input your text to build a Q&A Bot here.....")
          chatbot = gr.Chatbot()
          message = gr.Textbox ("know akhil?")
          message.submit(chat, [chatbot, message], chatbot)

demo.queue().launch(share=True)


"""# **Github Setup**"""



"""## Launch Phoenix

Define your knowledge base dataset with a schema that specifies the meaning of each column (features, predictions, actuals, tags, embeddings, etc.). See the [docs](https://docs.arize.com/phoenix/) for guides on how to define your own schema and API reference on `phoenix.Schema` and `phoenix.EmbeddingColumnNames`.
"""

# # get a random sample of 500 documents (including retrieved documents)
# # this will be handled by by the application in a coming release
# num_sampled_point = 500
# retrieved_document_ids = set(
#     [
#         doc_id
#         for doc_ids in query_df[":feature.[str].retrieved_document_ids:prompt"].to_list()
#         for doc_id in doc_ids
#     ]
# )
# retrieved_document_mask = database_df["document_id"].isin(retrieved_document_ids)
# num_retrieved_documents = len(retrieved_document_ids)
# num_additional_samples = num_sampled_point - num_retrieved_documents
# unretrieved_document_mask = ~retrieved_document_mask
# sampled_unretrieved_document_ids = set(
#     database_df[unretrieved_document_mask]["document_id"]
#     .sample(n=num_additional_samples, random_state=0)
#     .to_list()
# )
# sampled_unretrieved_document_mask = database_df["document_id"].isin(
#     sampled_unretrieved_document_ids
# )
# sampled_document_mask = retrieved_document_mask | sampled_unretrieved_document_mask
# sampled_database_df = database_df[sampled_document_mask]

# database_schema = px.Schema(
#     prediction_id_column_name="document_id",
#     prompt_column_names=px.EmbeddingColumnNames(
#         vector_column_name="text_vector",
#         raw_data_column_name="text",
#     ),
# )
# database_ds = px.Dataset(
#     dataframe=sampled_database_df,
#     schema=database_schema,
#     name="database",
# )

"""Define your query dataset. Because the query dataframe is in OpenInference format, Phoenix is able to infer the meaning of each column without a user-defined schema by using the `phoenix.Dataset.from_open_inference` class method."""

# query_ds = px.Dataset.from_open_inference(query_df)

"""Launch Phoenix. Follow the instructions in the cell output to open the Phoenix UI."""

# session = px.launch_app(primary=query_ds, corpus=database_ds)

