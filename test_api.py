import os
import sys
import json
import requests
from decouple import config

# Default Modal endpoint URL. Update this with your actual deployed Modal endpoint URL.
DEFAULT_API_URL = "https://lisakoeritz--ai-guidelines-guru-endpoint-dev.modal.run"

def get_auth_headers():
    # Load authentication token from environment or fallback to .env/local value
    try:
        token = config("AIEN_AUTH_TOKEN")
    except Exception:
        token = os.getenv("AIEN_AUTH_TOKEN", "")
    
    if not token:
        print("Warning: AIEN_AUTH_TOKEN is not set in environment or .env file.")
    
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def handle_response_error(status_code, text):
    print(f"Status Code: {status_code}")
    print(f"Error Response: {text}")
    if "invalid function call" in text or "modal-http" in text:
        print("\n[TIP] This error usually means your Modal app is not currently running.")
        print("To run it in development/serve mode, execute:")
        print("    poetry run modal serve src/app.py")
        print("To deploy it permanently to production, execute:")
        print("    poetry run modal deploy src/app.py")
        print("Make sure your API URL matches the deployed/served environment.")
        print("  - Serve environment (dev) URL usually contains '-dev.modal.run'")
        print("  - Deploy environment (prod) URL usually contains '.modal.run' (without '-dev')\n")

def test_qa_endpoint(api_url, question):
    url = f"{api_url.rstrip('/')}/api/qa"
    headers = get_auth_headers()
    data = {"question": question}
    
    print(f"\n--- Testing QA Endpoint: {url} ---")
    print(f"Question: '{question}'")
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            print(f"Status Code: {response.status_code}")
            result = response.json()
            print("\nResponse Answer:")
            print(result.get("answer"))
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result.get("documents", []), 1):
                metadata = doc.get("metadata", {})
                source = metadata.get("document_name", "Unknown")
                page = metadata.get("page", "N/A")
                link = metadata.get("document_url") or metadata.get("link") or "N/A"
                print(f"  [{i}] Source: {source} (Page: {page}, Link: {link})")
                print(metadata)
        else:
            handle_response_error(response.status_code, response.text)
            
    except Exception as e:
        print(f"Error connecting to API: {e}")

def test_qa_stream_endpoint(api_url, question):
    url = f"{api_url.rstrip('/')}/api/qa/stream"
    headers = get_auth_headers()
    data = {"question": question}
    
    print(f"\n--- Testing Streaming QA Endpoint: {url} ---")
    print(f"Question: '{question}'")
    print("\nStreaming response tokens:")
    
    try:
        # requests.post with stream=True to handle server-sent events (SSE)
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            if response.status_code != 200:
                print("") # newline
                handle_response_error(response.status_code, response.text)
                return
                
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        event_data = decoded_line[6:] # Strip "data: "
                        try:
                            event = json.loads(event_data)
                            event_type = event.get("event_type")
                            
                            if event_type == "on_retriever_end":
                                print("\n[Retrieved Documents]")
                                for i, doc in enumerate(event.get("content", []), 1):
                                    metadata = doc.get("metadata", {})
                                    source = metadata.get("document_name", "Unknown")
                                    page = metadata.get("page", "N/A")
                                    link = metadata.get("document_url") or metadata.get("link") or "N/A"
                                    print(f"  [{i}] Source: {source} (Page: {page}, Link: {link})")
                                print("\n[Streaming Answer Tokens]")
                                
                            elif event_type == "on_chat_model_stream":
                                token = event.get("content", "")
                                sys.stdout.write(token)
                                sys.stdout.flush()
                                
                            elif event_type == "done":
                                print("\n\n[Streaming finished]")
                                
                        except json.JSONDecodeError:
                            print(f"\nFailed to parse SSE event JSON: {decoded_line}")
                            
    except Exception as e:
        print(f"\nError during streaming connection: {e}")

if __name__ == "__main__":
    # Check if a custom URL is provided as argument, else use default Modal URL
    api_base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_URL
    test_question = "What are the core principles of AI transparency?"
    
    print(f"Targeting API Endpoint Base: {api_base_url}")
    
    # Run the tests
    test_qa_endpoint(api_base_url, test_question)
    test_qa_stream_endpoint(api_base_url, test_question)
