import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
#pandas, numpy — data

# langchain_* — LLM-based vector search

# gradio — UI

# dotenv — to load OpenAI API keys from .env




load_dotenv()
#Loads your OpenAI API key from .env.

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)
#Adds high-res book cover URLs, falling back to "cover-not-found.jpg" if missing.
#Build Vector DB for Semantic Search
#Loads and splits book descriptions into chunks for vectorization.

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
#Builds a Chroma vector store from the documents using OpenAI embeddings. This will be used to find books semantically similar to a user's query.
#Retrieve Recommendations by Semantics + Filters
def retrieve_semantic_recommendations(#Use semantic similarity to find similar books.

#Filter by category and emotion tone.
        
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    #Searches vector DB for top 50 books most similar to the query.
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    #Extracts book IDs (ISBNs) from the matched documents.
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    #Gets book metadata (title, author, etc.) from books DataFrame using matched ISBNs.
#Narrows down to books of the selected category.
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs
#If a tone (emotion) is selected, sort books based on that emotion score in descending order.

#Gets final filtered list and builds a list of (image, caption) for display.

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
#Shortens the book description for display.
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
#Formats author names nicely.
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results
#Returns a list of (image, caption) to render in the UI gallery.
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
#Dropdown options for category and emotional tone.
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")
#Creates a beautiful Gradio UI using the Glass theme.
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")
# Collects user input for:

# Free-text query (semantic)

# Category filter

# Emotional tone filter
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)
#Where the recommended books will be displayed as thumbnails with captions.


    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)
#Connects the recommend_books() function to the button.

if __name__ == "__main__":
    dashboard.launch()
    #Launches the Gradio app in your browser.
    gr.close_all()
dashboard.launch(share=True)