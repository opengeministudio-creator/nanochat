import json

from tasks.customjson import CustomJSON


def test_customjson_allows_optional_system_role(tmp_path):
    fpath = tmp_path / "conversations.jsonl"
    conversation = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello"},
    ]
    fpath.write_text(json.dumps(conversation) + "\n", encoding="utf-8")

    dataset = CustomJSON(filepath=str(fpath))
    assert dataset.num_examples() == 1
    assert dataset.get_example(0)["messages"][0]["role"] == "system"
