import json

from volume2gym.exporters import grpo_record, sft_record, write_sft_jsonl
from volume2gym.models import AnswerKey, Citation, Task, TaskFamily


def make_task():
    citation = Citation(document_id="manual", span_id="span-1")
    return Task(
        task_id="task-1",
        prompt="What is required?",
        task_family=TaskFamily.STANDARD_OPERATION,
        knowledge_unit_ids=("unit-1",),
        answer_key=AnswerKey(
            applicable_unit_ids=("unit-1",),
            required_actions=("stop",),
            citations=(citation,),
        ),
        citations=(citation,),
    )


def test_exports_keep_answer_key_private_from_prompt():
    task = make_task()
    sft = sft_record(task)
    grpo = grpo_record(task)

    assert sft["messages"][1]["content"] == task.prompt
    assert json.loads(sft["messages"][2]["content"])["required_actions"] == ["stop"]
    assert "answer_key" in grpo
    assert "stop" not in grpo["prompt"]


def test_sft_writer_is_deterministic(tmp_path):
    left = write_sft_jsonl([make_task()], tmp_path / "left.jsonl")
    right = write_sft_jsonl([make_task()], tmp_path / "right.jsonl")
    assert left.read_bytes() == right.read_bytes()
