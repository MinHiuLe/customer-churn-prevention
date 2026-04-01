import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serving.main import app

@pytest.fixture(scope="session")
def client():
    """TestClient với lifespan context — models được load đúng cách"""
    with TestClient(app) as c:
        yield c
