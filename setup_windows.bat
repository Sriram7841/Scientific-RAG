@echo off
REM ================================================================
REM  GraphRAG Scientific Research Assistant — Windows CMD Setup
REM  Python 3.10 compatible
REM  Usage: Double-click this file OR run in CMD: setup_windows.bat
REM ================================================================

echo.
echo  ============================================================
echo   GraphRAG Scientific Research Assistant - Windows Setup
echo   arXiv 206K Papers - 25 Categories - SVM + FAISS + LLM
echo  ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Download Python 3.10 from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [1/6] Python detected:
python --version
echo.

REM Create virtual environment
echo  [2/6] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo  ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo  Virtual environment created.
echo.

REM Activate
call venv\Scripts\activate.bat
echo  [3/6] Installing packages (takes 5-15 min first time)...

REM Upgrade pip first
python -m pip install --upgrade pip setuptools wheel --quiet

REM PyTorch CPU-only
echo  Installing PyTorch (CPU only)...
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo  Trying alternate PyTorch install...
    pip install torch==2.2.2 --quiet
)

REM Rest of requirements
echo  Installing all other packages...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    pip install -r requirements.txt
)
echo  All packages installed.
echo.

REM NLTK
echo  [4/6] Downloading NLTK stopwords...
python -c "import nltk; nltk.download('stopwords', quiet=True)"
echo.

REM Environment
echo  [5/6] Setting up .env file...
if not exist .env (
    copy .env.example .env
    echo  Created .env from .env.example
    echo  IMPORTANT: Edit .env and add your HF_API_KEY for AI answers
) else (
    echo  .env already exists
)
echo.

REM Directories + DB
echo  [6/6] Setting up database...
if not exist data mkdir data
if not exist models mkdir models
if not exist media mkdir media
if not exist staticfiles mkdir staticfiles

python manage.py makemigrations accounts rag_engine chat --noinput 2>nul
python manage.py migrate --noinput
python manage.py create_admin
echo.

echo  ============================================================
echo   Setup Complete!
echo  ============================================================
echo.
echo  NEXT STEPS:
echo.
echo  1. Place your dataset:
echo       data\arxiv.csv   (columns: title, abstract, category)
echo.
echo  2. Activate venv in new CMD windows:
echo       venv\Scripts\activate
echo.
echo  3. Run ML pipeline:
echo       python manage.py setup_graphrag
echo.
echo     Fast test (20K rows):
echo       python manage.py setup_graphrag --rows 20000
echo.
echo  4. Start server:
echo       python manage.py runserver
echo.
echo  5. Open browser:
echo       http://127.0.0.1:8000
echo.
echo  Admin: admin / admin123
echo.
pause
