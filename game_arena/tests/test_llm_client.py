from app.llm_client import parse_ai_response

def test_parse_valid_json():
    text = '{"inner_monologue": "Think", "public_speech": "Hello"}'
    res = parse_ai_response(text)
    assert res["inner_monologue"] == "Think"
    assert res["public_speech"] == "Hello"

def test_parse_json_with_extra_text():
    text = 'Random prefix {"inner_monologue": "Thoughts", "public_speech": "Speech"} and suffix'
    res = parse_ai_response(text)
    assert res["inner_monologue"] == "Thoughts"
    assert res["public_speech"] == "Speech"

def test_parse_fallback_to_plain_text():
    text = "Just a plain text response without JSON"
    res = parse_ai_response(text)
    assert res["inner_monologue"] == "(Thinking...)"
    assert res["public_speech"] == text

def test_parse_empty():
    res = parse_ai_response("")
    assert res["inner_monologue"] == "(No response)"
    assert res["public_speech"] == "(No response)"
