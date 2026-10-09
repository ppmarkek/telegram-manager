import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Клиент Anthropic создаётся при импорте App; настоящий ключ в тестах не нужен.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
