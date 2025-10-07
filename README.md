# Microcontroller RAG + LINE Bot (Run locally in VS Code)

This repository contains a RAG-based chatbot originally run in Google Colab. The code has been updated to run in VS Code (or any local Python environment) using environment variables and python-dotenv.

## Quick setup

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your API keys and tokens:

- `GOOGLE_API_KEY` - Google GenAI API key
- `COHERE_API_KEY` - (optional) Cohere API key for reranking
- `LINE_CHANNEL_ACCESS_TOKEN` and `LINE_CHANNEL_SECRET` - LINE bot credentials
- `NGROK_AUTHTOKEN` - ngrok auth token (optional, required if you want public webhook via ngrok)

4. Run the bot:

```powershell
python app.py
```

Optional: use the included VS Code launch configuration to run `app.py` with your `.env` loaded.

The script will start Flask on port 5000 and create an ngrok tunnel. It prints the public webhook URL for LINE. Configure your LINE Messaging API webhook to point to `<public-url>/callback`.

## Notes

- The code now reads secrets from environment variables. You can still keep the hardcoded values in `code.py` for quick tests, but using `.env` is strongly recommended.
- If you encounter missing-package errors, install the missing package individually (often package names in this project map to multiple LangChain community packages).

If you'd like, I can also create a VS Code launch configuration or automate creating the virtual environment and activating it.
