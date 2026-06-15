import requests
import aiohttp
import asyncio
import re
import os
from decouple import config

import pandas as pd
import json

from llama_parse import LlamaParse
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_metadata_csv(metadata_path):
    metadata = pd.read_csv(metadata_path, delimiter=';')
    return metadata

def download_documents(metadata, output_dir='documents', start_id=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, row in metadata.iloc[start_id:].iterrows():
        url = row['document_url']
        id = row['document_id']
        #check if the file already exists
        if os.path.exists(f"{output_dir}/document_{id}.pdf"):
            print(f"File document_{id}.pdf already exists")
            continue
        try:
            # Fetch the content from the URL
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Determine content type and file extension
            content_type = response.headers.get('Content-Type')
            if 'application/pdf' in content_type:
                file_extension = '.pdf'
            else:
                file_extension = '.html'
            
            # Generate a filename using the index of the URL
            filename = os.path.join(output_dir, f'document_{id}{file_extension}')
            
            # Save the file
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded {id} -- {url} as {filename}")
        
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {id} -- {url}: {e}")

def parse_html(html_content_path):
    loader = UnstructuredHTMLLoader(html_content_path)
    docs = loader.load()
    return docs

def parse_pdf(pdf_content_path):
    loader = PDFPlumberLoader(pdf_content_path)
    docs = loader.load()
    return docs

# use langchain to split pdf text into chunks
def chunk_document(doc, metadata):
    """Split a document into chunks with metadata."""
    if doc is None:
        print("Warning: Received None as document for chunking.")
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    try:
        all_splits = text_splitter.split_documents(doc)
    except Exception as e:
        print(f"Error splitting document: {e}")
        return []

    # Add json metadata to each chunk
    for d in all_splits:
        for k, v in metadata.items():
            d.metadata[k] = v
    
    return all_splits

def parse_and_split_downloaded_documents(dirpath='data/documents', metadata_path='data/registered_documents.json'):
    """Parse documents and split into chunks."""
    docs = []
    # load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Metadata file {metadata_path} not found.")
        return []

    if not os.path.exists(dirpath):
        return []

    for filename in os.listdir(dirpath):
        # Skip hidden files
        if filename.startswith('.'):
            continue
            
        file_path = os.path.join(dirpath, filename)
        
        # Skip empty files
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue
            
        print(f"DEBUG: Processing {filename}...")
        
        doc = None
        if filename.endswith('.pdf'):
            doc = parse_pdf(file_path)
        elif filename.endswith('.html'):
            doc = parse_html(file_path)
        else:
            continue

        if not doc:
            print(f"DEBUG: Document {filename} resulted in empty/None parse.")
            continue

        # Extract ID and split
        try:
            doc_id = filename.split(".")[0].split("_")[-1]
            if doc_id in metadata:
                doc_chunks = chunk_document(doc, metadata[doc_id])
                docs.extend(doc_chunks)
            else:
                print(f"DEBUG: No metadata entry for {doc_id}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return docs

# check if file is a pdf or html
async def fetch_contenttype_from_url(session, url):
    try:
        async with session.get(url, timeout=360) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type')
            if 'application/pdf' in content_type:
                return 'pdf', url
            else:
                return 'html', url
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return 'error', url

async def get_pdf_html_urls(url_list):
    '''
    Separate the provided list of urls strings into lists of urls that lead to pdfs and htmls

    Args:
    url_list (list): The list of urls to separate

    Returns:
    list: The list of pdf urls
    list: The list of html urls
    list: The list of urls that failed to download
    '''
    pdf_urls = []
    html_urls = []
    error_urls = []
    pdf_pattern = re.compile(r'(/|%2F|=)([^/]+\.pdf)(\?|&|$)', re.IGNORECASE)
    for url in url_list:
        if pdf_pattern.search(url):
            pdf_urls.append(url)
        else:
            html_urls.append(url)
    # make sure html urls are not pdfs
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_contenttype_from_url(session, url) for url in html_urls]
        results = await asyncio.gather(*tasks)

        for result_type, url in results:
            if result_type == 'pdf':
                pdf_urls.append(url)
                #remove the url from the html urls
                html_urls.remove(url)
            elif result_type == 'error':
                error_urls.append(url)
                html_urls.remove(url)

    return pdf_urls, html_urls, error_urls
            




