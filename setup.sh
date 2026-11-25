#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  GraphRAG Scientific Research Assistant — Automated Setup Script
#  Tested on: Windows 10/11 (Git Bash / WSL), macOS, Ubuntu
#  Python: 3.10+
#  Usage:  bash setup.sh
# ═══════════════════════════════════════════════════════════════════
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✅  $*${NC}"; }
info() { echo -e "${CYAN}  ℹ️   $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠️   $*${NC}"; }
err()  { echo -e "${RED}  ❌  $*${NC}"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  GraphRAG Scientific Research Assistant — Setup          ║${NC}"
echo -e "${CYAN}║  arXiv 206K Papers · 25 Categories · SVM + FAISS        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check Python 3.10 ────────────────────────────────────────────
command -v python3 &>/dev/null || command -v python &>/dev/null || \
  err "Python not found. Install Python 3.10 from https://www.python.org/downloads/"

PY=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
info "Detected: $PY"
PYBIN="python3"
command -v python3 &>/dev/null || PYBIN="python"

# ── Step 1: Virtual environment ───────────────────────────────────
echo ""
echo "  [1/6] Creating virtual environment …"
$PYBIN -m venv venv || err "Failed to create venv. Run: pip install virtualenv"

# Activate (cross-platform)
if [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate   # Windows Git Bash
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate        # Linux / macOS / WSL
else
  err "Cannot activate virtual environment."
fi
ok "Virtual environment active"

# ── Step 2: Upgrade pip ───────────────────────────────────────────
echo ""
echo "  [2/6] Upgrading pip …"
pip install --upgrade pip setuptools wheel --quiet
ok "pip upgraded"

# ── Step 3: Install dependencies ─────────────────────────────────
echo ""
echo "  [3/6] Installing Python packages (may take 5-15 min first time) …"
info "Installing PyTorch CPU-only …"
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null || \
pip install torch==2.2.2 --quiet 2>/dev/null || \
warn "PyTorch install issue — trying requirements.txt fallback"

info "Installing all other requirements …"
pip install -r requirements.txt --quiet 2>/dev/null || \
pip install -r requirements.txt
ok "All packages installed"

# ── Step 4: NLTK data ─────────────────────────────────────────────
echo ""
echo "  [4/6] Downloading NLTK stopwords …"
python -c "import nltk; nltk.download('stopwords', quiet=True); print('  NLTK stopwords ready')" || \
warn "NLTK download failed (non-fatal — built-in fallback will be used)"

# ── Step 5: Environment file ──────────────────────────────────────
echo ""
echo "  [5/6] Setting up environment …"
if [ ! -f ".env" ]; then
  cp .env.example .env
  ok ".env created from .env.example"
  warn "Edit .env and set HF_API_KEY for LLM-generated answers"
  warn "Without HF_API_KEY the system still works (uses extractive answers)"
else
  ok ".env already exists"
fi

# ── Step 6: Directory + DB ────────────────────────────────────────
echo ""
echo "  [6/6] Creating directories and database …"
mkdir -p data models media staticfiles

python manage.py makemigrations accounts rag_engine chat --noinput 2>/dev/null || true
python manage.py migrate --noinput
ok "Database migrated (db.sqlite3)"

python manage.py create_admin
ok "Admin user ready"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅  Setup Complete!                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}NEXT STEPS:${NC}"
echo ""
echo "  1. Place your dataset:"
echo "       data/arxiv.csv   ← columns: title, abstract, category"
echo "       (Download: https://www.kaggle.com/datasets)"
echo ""
echo "  2. Activate virtual environment (every new terminal):"
echo "       Windows CMD:   venv\\Scripts\\activate"
echo "       Windows PS:    venv\\Scripts\\Activate.ps1"
echo "       Linux/macOS:   source venv/bin/activate"
echo ""
echo "  3. Run the full ML pipeline (train + index + graph):"
echo "       python manage.py setup_graphrag"
echo ""
echo "     Or step by step:"
echo "       python manage.py setup_graphrag --rows 20000        # fast test"
echo "       python manage.py setup_graphrag --skip-index        # train only"
echo "       python manage.py setup_graphrag --skip-train        # index only"
echo ""
echo "  4. Start the server:"
echo "       python manage.py runserver"
echo ""
echo "  5. Open in browser:"
echo "       http://127.0.0.1:8000"
echo ""
echo "  Admin login:  admin  /  admin123"
echo ""
echo "  REST API:"
echo "    POST /api/accounts/token/   → get JWT token"
echo "    POST /api/chat/query/       → chat via API"
echo "    POST /api/rag/predict/      → classify a query"
echo ""
