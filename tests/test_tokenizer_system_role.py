import pytest

pytest.importorskip("tokenizers")

from nanochat.tokenizer import HuggingFaceTokenizer


def _make_test_tokenizer():
    text_corpus = [
        "system user assistant",
        "You are helpful",
        "hello world",
        "hi there",
    ]
    return HuggingFaceTokenizer.train_from_iterator(text_corpus, vocab_size=512)


def test_render_conversation_supports_optional_system_role():
    tokenizer = _make_test_tokenizer()
    conversation = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    }

    ids, mask = tokenizer.render_conversation(conversation)
    decoded = tokenizer.decode(ids)

    assert "system" in decoded
    assert "user" in decoded
    assert "assistant" in decoded
    assert sum(mask) > 0
