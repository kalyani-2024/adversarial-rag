import pytest
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import LLMOutputError
from app.core.llm import parse_json_model
from app.schemas.query import QueryRequest


class _Model(BaseModel):
    a: int


def test_parse_json_plain():
    assert parse_json_model('{"a": 1}', _Model).a == 1


def test_parse_json_with_fences_and_prose():
    text = 'Sure! Here you go:\n```json\n{"a": 2}\n```'
    assert parse_json_model(text, _Model).a == 2


def test_parse_json_invalid_raises():
    with pytest.raises(LLMOutputError):
        parse_json_model("no json here", _Model)
    with pytest.raises(LLMOutputError):
        parse_json_model('{"a": "not-int"}', _Model)


def test_settings_reject_overlap_ge_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)


def test_settings_secret_not_in_repr():
    s = Settings(_env_file=None, groq_api_key="gsk_supersecret")
    assert "gsk_supersecret" not in repr(s)


def test_query_request_strips_and_rejects_blank():
    assert QueryRequest(query="  hi  ").query == "hi"
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")
