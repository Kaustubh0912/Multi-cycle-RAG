# Commands.md

## CLI Commands Reference

### Overview
This CLI tool provides a professional interface for the Advanced RAG (Retrieval-Augmented Generation) Engine with GitHub Models integration.

---

## Available Commands

### 1. `ingest`
**Purpose:** Ingest documents into the vector store for retrieval.

**Usage:**
```bash
python cli.py ingest [directory]
```

**Parameters:**
- `directory` (optional): The directory containing documents to ingest
  - Default: `./docs`
  - Type: String

**Description:**
- Processes and ingests documents from the specified directory
- Creates embeddings and stores them in the ChromaDB vector database
- Displays progress with a spinner and success/error messages
- Returns the number of document chunks successfully ingested

**Example:**
```bash
python cli.py ingest ./my-documents
python cli.py ingest  # Uses default ./docs directory
```

---

### 2. `delete-db`
**Purpose:** Delete the ChromaDB database directory permanently.

**Usage:**
```bash
python cli.py delete-db
```

**Parameters:** None

**Description:**
- Prompts for confirmation before deletion
- Removes the entire ChromaDB persistence directory
- Shows appropriate messages for success, warnings, or errors
- Cannot be undone - use with caution

**Example:**
```bash
python cli.py delete-db
# Prompts: "Are you sure you want to permanently delete the database at '[path]'? (yes/no) [no]:"
```

---

### 3. `test`
**Purpose:** Run automated tests for the RAG engine functionality.

**Usage:**
```bash
python cli.py test
```

**Parameters:** None

**Description:**
- Executes a comprehensive test suite for the RAG engine
- Tests include:
  1. Document ingestion from `./docs`
  2. Simple RAG query testing
  3. Decomposed RAG query testing (complex queries)
- Displays results in formatted tables
- Shows test progress with spinners and status updates

**Example:**
```bash
python cli.py test
```

---

### 4. `chat` (Default Command)
**Purpose:** Start an interactive chat session with the RAG engine.

**Usage:**
```bash
python cli.py chat [options]
python cli.py  # Defaults to chat mode
```

**Parameters:**
- `-d, --decomposition`: Enable query decomposition mode for complex questions
- `--no-stream`: Disable streaming responses (wait for full response)

**Description:**
- Launches an interactive chat interface
- Automatically ingests documents from `./docs` at startup
- Supports two modes:
  - **Simple Mode**: Direct question-answering
  - **Decomposition Mode**: Breaks complex questions into sub-queries
- Features streaming responses by default
- Type `exit` or `quit` to end the session
- Handles keyboard interrupts gracefully

**Examples:**
```bash
python cli.py chat                    # Simple mode with streaming
python cli.py chat -d                 # Decomposition mode with streaming
python cli.py chat --no-stream        # Simple mode without streaming
python cli.py chat -d --no-stream     # Decomposition mode without streaming
python cli.py                         # Defaults to simple chat mode
```

---

## Command Execution Notes

### Default Behavior
- If no command is specified, the CLI defaults to `chat` mode
- All commands use Rich library for enhanced terminal UI with colors, panels, and progress indicators

### Error Handling
- All commands include comprehensive error handling
- Errors are displayed in formatted panels with appropriate styling
- Failed operations show detailed error messages

### Asynchronous Operations
- All commands run asynchronously for better performance
- Progress is shown with animated spinners during long-running operations

### Dependencies
- Requires the RAG engine to be properly configured
- Uses ChromaDB for vector storage
- Integrates with GitHub Models for LLM functionality
