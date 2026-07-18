import pytest

from volume2gym.providers import (
    CallableGenerator,
    GenerationRequest,
    ProviderError,
    parse_json_object,
)


def test_parse_json_object_accepts_fenced_json():
    assert parse_json_object('```json\n{"rules": []}\n```') == {"rules": []}


def test_parse_json_object_rejects_arrays():
    with pytest.raises(ProviderError, match="not an object"):
        parse_json_object("[]")


def test_callable_generator_preserves_failures():
    def fail(_request):
        raise RuntimeError("offline")

    generator = CallableGenerator(fail, provider="fixture", model="broken")
    request = GenerationRequest(system="system", prompt="prompt", schema_name="rules")
    with pytest.raises(ProviderError, match="fixture/broken failed: offline"):
        generator.generate(request)
