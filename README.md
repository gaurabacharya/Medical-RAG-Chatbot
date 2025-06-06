# Medical-RAG-Chatbot

## Demo of the API working with a frontend UI
https://youtu.be/-NcgFqA0GN8

## Summary
This medical chatbot was created using RAG (Retrieval Augmented Generation) to produce 
more correct responses than the LLM model by providing context data from a vectorized database that can be updated with more recent data. This extra context reduces hallucinations in responses, the system prompt creates consistent responses, and citations to where the top 3 sources that provided the context are also provided in the response. Although there is html code for a simple UI, the main purpose is to create an backend API endpoint using Flask to be used in another frontend program with a better UI which is in another repository [here](https://github.com/gaurabacharya/medical-chatbot-frontend). The data that was provided was the Gale Encyclopedia of Medicine and the data was split and embedded using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from hugging face. The vectorized data was then stored on [PineCone](https://www.pinecone.io/). Used [langchain](https://python.langchain.com/docs/introduction/) to create documents, split, create the retrieval chain to retrieve the context from Pinecone, and apply the prompt to our LLM. Lastly, the LLM that was used was [gemini-2.0-flash](https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash-exp). 

<img width="917" alt="Medical Chatbot" src="https://github.com/user-attachments/assets/462afa80-f6db-4ffc-b2f3-32ecbe75ff70" />

## How to Run 

### Step 1: Copy the repo and create a virtual environment 
```bash
python3.10 -m venv llmapp
```

Activate it

Mac
```bash
source ./llmapp/bin/activate
```

Windows
```bash
 env/Scripts/activate.bat // in CMD
 ```

### Step 2: Install the requirements 
```bash
pip install -r requirements.txt
```

### Step 3: Obtain API Keys and store them in .env file 
```base
PINECONE_API_KEY = "..."
GOOGLE_API_KEY = "..."
```
[PineCone](https://www.pinecone.io/)
[Google API Key](https://ai.google.dev/gemini-api/docs/api-key)


### Step 4: Run store_index.py to upload data to PineCone
```bash
cd <working directory>
python store_index.py
```

### Step 5: Run app.py locally
```bash
python app.py
```

Now you can ask medical question to the bot! 
