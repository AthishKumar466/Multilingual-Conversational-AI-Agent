# Multilingual Chatbot

## Quick start

1. Create a virtual environment and install requirements:

```bash
pip install -r backend/requirements.txt
```

2. Export your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

3. Run the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open [http://localhost:8000](http://localhost:8000).

