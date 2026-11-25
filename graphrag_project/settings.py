import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY       = os.getenv('SECRET_KEY', 'django-insecure-graphrag-dev-key-2024')
DEBUG            = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS    = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework_simplejwt',
    'corsheaders',
    'accounts.apps.AccountsConfig',
    'admin_panel.apps.AdminPanelConfig',
    'rag_engine.apps.RagEngineConfig',
    'chat.apps.ChatConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'graphrag_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'graphrag_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE      = 'UTC'
USE_I18N       = True
USE_TZ         = True

STATIC_URL       = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT      = BASE_DIR / 'staticfiles'

MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
AUTH_USER_MODEL    = 'accounts.User'

# ── REST + JWT ────────────────────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
}
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME':  timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'AUTH_HEADER_TYPES': ('Bearer',),
}
CORS_ALLOWED_ORIGINS = ['http://localhost:8000', 'http://127.0.0.1:8000']
CORS_ALLOW_CREDENTIALS = True

# ── File paths ────────────────────────────────────────────────────
DATA_DIR         = BASE_DIR / 'data'
MODELS_DIR       = BASE_DIR / 'models'
FAISS_INDEX_PATH = BASE_DIR / 'data' / 'faiss_index'
GRAPH_PATH       = BASE_DIR / 'data' / 'knowledge_graph.pkl'
SVM_MODEL_PATH   = BASE_DIR / 'models' / 'svm_model.pkl'
NB_MODEL_PATH    = BASE_DIR / 'models' / 'nb_model.pkl'
TFIDF_PATH       = BASE_DIR / 'models' / 'tfidf_vectorizer.pkl'
LABEL_ENC_PATH   = BASE_DIR / 'models' / 'label_encoder.pkl'

# ── AI settings ───────────────────────────────────────────────────
# ── AI settings ───────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv(
    'EMBEDDING_MODEL',
    'sentence-transformers/all-MiniLM-L6-v2'
)

# =========================================================
# HUGGING FACE API
# =========================================================

HF_API_KEY = os.getenv('HF_API_KEY', '')

# =========================================================
# WORKING LIGHTWEIGHT HF MODELS
# =========================================================

# Primary generation model
models = [
    "gpt2",
    "distilgpt2",
]

# =========================================================
# DATASET LIMIT
# =========================================================

MAX_DATASET_ROWS = int(
    os.getenv('MAX_DATASET_ROWS', '30000')
)

# 25 arXiv categories in this dataset
ARXIV_CATEGORIES = {
    'cs.LG':    'Machine Learning',
    'cs.CV':    'Computer Vision',
    'cs.CL':    'Natural Language Processing',
    'quant-ph': 'Quantum Physics',
    'cs.RO':    'Robotics',
    'cs.AI':    'Artificial Intelligence',
    'cs.NE':    'Neural & Evolutionary Computing',
    'cs.IR':    'Information Retrieval',
    'cs.CR':    'Cryptography & Security',
    'cs.DS':    'Data Structures & Algorithms',
    'cs.DC':    'Distributed Computing',
    'cs.SE':    'Software Engineering',
    'cs.PL':    'Programming Languages',
    'cs.DB':    'Databases',
    'cs.HC':    'Human-Computer Interaction',
    'cs.MA':    'Multiagent Systems',
    'cs.GT':    'Game Theory',
    'cs.SY':    'Systems & Control',
    'math.ST':  'Statistics Theory',
    'math.OC':  'Optimization & Control',
    'math.PR':  'Probability',
    'stat.ML':  'Statistics - Machine Learning',
    'stat.ME':  'Statistics - Methodology',
    'physics.comp-ph': 'Computational Physics',
    'cond-mat.dis-nn': 'Disordered Systems & Neural Networks',
}

# ── Cache ─────────────────────────────────────────────────────────
REDIS_URL = os.getenv('REDIS_URL', '')
if REDIS_URL:
    CACHES = {
        'default': {
            'BACKEND':  'django.core.cache.backends.redis.RedisCache',
            'LOCATION': REDIS_URL,
            'TIMEOUT':  300,
        }
    }
else:
    CACHES = {
        'default': {
            'BACKEND':  'django.core.cache.backends.locmem.LocMemCache',
            'LOCATION': 'graphrag-cache',
            'TIMEOUT':  300,
        }
    }

LOGIN_URL           = '/accounts/login/'
LOGIN_REDIRECT_URL  = '/chat/'
LOGOUT_REDIRECT_URL = '/accounts/login/'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'colored': {
            'format': '\033[36m[%(levelname)s]\033[0m %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'colored'},
    },
    'root': {'handlers': ['console'], 'level': 'INFO'},
    'loggers': {
        'rag_engine': {'handlers': ['console'], 'level': 'DEBUG', 'propagate': False},
    },
}
