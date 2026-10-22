"""Regression tests for NEXA-423 — a palace that could not be read must be a
failed tool call, not an empty answer.

``mempalace_search`` answers a broken palace with ``{"results": [],
"error": ...}`` and ``mempalace_diary_read`` with ``{"error": ...}``. An agent
that reads the payload but not the error field concludes the shared memory is
empty and works on a false premise (NEXA-423). MCP's ``isError`` flag is the
channel for that: it marks the call as failed without changing the payload,
so nothing that used to be answered stops being answered.
"""

from _mcp_server_helpers import _patch_mcp_server


def _call(monkeypatch, config, palace_path, kg, tool_name, arguments):
    _patch_mcp_server(monkeypatch, config, kg)

    from mempalace.mcp_server import handle_request

    return handle_request(
        {
            "method": "tools/call",
            "id": 1,
            "params": {"name": tool_name, "arguments": arguments},
        }
    )


def test_storage_failure_is_marked_as_a_failed_tool_call(monkeypatch, config, palace_path, kg):
    """A palace that cannot be opened answers with ``isError``."""
    response = _call(
        monkeypatch,
        config,
        palace_path,
        kg,
        "mempalace_diary_read",
        {"agent_name": "TestAgent", "last_n": 5},
    )

    assert "error" not in response, "the call itself must not raise"
    result = response["result"]
    assert result.get("isError") is True, (
        f"a palace that could not be read must be a failed tool call, not an empty answer: {result}"
    )
    assert "content" in result, "the payload must still be delivered"


def test_successful_read_is_not_marked_as_a_failure(monkeypatch, config, palace_path, kg):
    """A read that worked must not be marked as a failure."""
    _patch_mcp_server(monkeypatch, config, kg)

    from mempalace.mcp_server import handle_request, tool_diary_write

    written = tool_diary_write(
        agent_name="TestAgent",
        entry="A successful read must stay a successful read.",
        topic="general",
    )
    assert written["success"] is True

    response = handle_request(
        {
            "method": "tools/call",
            "id": 2,
            "params": {
                "name": "mempalace_diary_read",
                "arguments": {"agent_name": "TestAgent", "last_n": 5},
            },
        }
    )

    assert "error" not in response
    assert response["result"].get("isError") is not True
    assert response["result"]["content"], "the payload must be delivered"
