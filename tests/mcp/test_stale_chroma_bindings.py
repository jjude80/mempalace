"""Regression tests for a chroma client whose rust bindings were
deleted by a stopped chroma ``System``.

``chromadb``'s ``RustBindingsAPI.stop()`` runs ``del self.bindings``. A client
that outlived the ``System`` it was opened on keeps the object but loses the
attribute, and every later call on it raises::

    AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'

The MCP **session** layer caches both a client and the collection opened on
it, so a stopped ``System`` leaves both handles dead. Rebuilding only on a
disk change misses this: an idle palace's ``chroma.sqlite3`` does not move.
The diary, status and duplicate-check tools therefore stayed dark until
somebody ran a manual ``mempalace_reconnect``.

These tests reproduce the stopped-``System`` state on an idle palace and
assert that both the search path and the diary path recover on their own.
"""

from _mcp_server_helpers import _patch_mcp_server


def _stop_shared_system(client) -> None:
    """Stop the chroma ``System`` the client is bonded to, the way chroma does.

    Releasing the shared ``System``'s last reference pops it from chroma's
    map and calls ``System.stop()``, which deletes the rust bindings. Any
    client — and any collection opened on it — that outlives that ``System``
    is now dead.

    The releases are unconditional rather than counted: the session client and
    the backend client for one palace path each hold two references (the
    client plus its ``AdminClient``), and the exact count is chroma's business,
    not the test's.
    """
    from chromadb.api.shared_system_client import SharedSystemClient

    identifier = client._identifier
    for _ in range(8):
        SharedSystemClient._release_system(identifier)

    server = getattr(client, "_server", None)
    assert server is not None, "the client must be a chroma client"
    assert not hasattr(server, "bindings"), (
        "precondition: a stopped chroma System deletes its rust bindings"
    )


def _settle(monkeypatch, config, palace_path, kg):
    """Patch the MCP server and leave it holding a live, up-to-date handle.

    ``_get_client()`` rebuilds the client whenever ``chroma.sqlite3`` changed
    on disk, which masks a stopped ``System``: the rebuild hands back a fresh
    client and the failure never shows. A palace that nobody writes to is the
    case that breaks in production, so the test settles the freshness stamp
    first and stops the ``System`` afterwards.
    """
    _patch_mcp_server(monkeypatch, config, kg)

    from mempalace import mcp_server

    return mcp_server


def _mark_palace_unchanged(mcp_server, palace_path) -> None:
    """Record the palace's current stat as the one the client was built on.

    ``_get_client()`` rebuilds the client when ``chroma.sqlite3`` changed on
    disk, so a write that lands after the client was opened masks a stopped
    ``System`` behind a rebuild. On an idle palace — the state the hub is in
    when it breaks — the stat still matches and the dead client is served.
    """
    import os

    stat = os.stat(os.path.join(palace_path, "chroma.sqlite3"))
    mcp_server._palace_db_inode = stat.st_ino
    mcp_server._palace_db_mtime = stat.st_mtime


def test_session_layer_rebuilds_client_after_stopped_system(monkeypatch, config, palace_path, kg):
    """The session layer must notice the dead client and reopen the palace."""
    mcp_server = _settle(monkeypatch, config, palace_path, kg)

    # Build and cache the session client and its collection, the way a diary
    # write does.
    assert mcp_server._get_collection(create=True) is not None
    _mark_palace_unchanged(mcp_server, palace_path)
    dead_client = mcp_server._client_cache
    assert dead_client is not None

    _stop_shared_system(dead_client)

    # The next session-layer read must rebuild the client on a fresh System
    # instead of handing back the dead bindings.
    collection = mcp_server._get_collection()
    assert collection is not None, "the session layer must reopen the palace"
    assert mcp_server._client_cache is not dead_client, (
        "the dead client must be dropped, not reused"
    )
    assert hasattr(mcp_server._client_cache._server, "bindings")


def test_diary_read_recovers_after_stopped_system(monkeypatch, config, palace_path, kg):
    """``mempalace_diary_read`` must recover without a manual reconnect."""
    mcp_server = _settle(monkeypatch, config, palace_path, kg)

    from mempalace.mcp_server import tool_diary_read, tool_diary_write

    written = tool_diary_write(
        agent_name="TestAgent",
        entry="The hub is the only place the team can read yesterday from.",
        topic="general",
    )
    assert written["success"] is True

    # Re-baseline the freshness stamp after the write, so the failure below is
    # the stopped System and nothing else.
    assert mcp_server._get_collection() is not None
    _mark_palace_unchanged(mcp_server, palace_path)
    _stop_shared_system(mcp_server._client_cache)

    read = tool_diary_read(agent_name="TestAgent", last_n=5)
    assert "error" not in read, f"diary read must recover, got: {read.get('error')}"
    assert read["entries"], "the entry written before the failure must come back"
    assert "yesterday" in read["entries"][0]["content"]


def test_search_recovers_after_stopped_system(monkeypatch, config, palace_path, kg):
    """``mempalace_search`` must recover without a manual reconnect."""
    mcp_server = _settle(monkeypatch, config, palace_path, kg)

    from mempalace.mcp_server import tool_add_drawer, tool_search

    filed = tool_add_drawer(
        content="Deployment notes: the hub restarts itself when chroma drops its bindings.",
        wing="wing_testagent",
        room="general",
    )
    assert "error" not in filed

    assert mcp_server._get_collection() is not None
    _mark_palace_unchanged(mcp_server, palace_path)
    _stop_shared_system(mcp_server._client_cache)

    found = tool_search("hub restarts itself chroma bindings", limit=5)
    assert "error" not in found, f"search must recover, got: {found.get('error')}"
    assert found["results"], "the drawer filed before the failure must come back"


def test_collection_open_retries_when_system_stops_after_client_check(tmp_path, monkeypatch):
    """A ``System`` stopped between ``_client()``'s check and the open retries.

    ``_client()`` rejects a cached client whose bindings are already gone, but
    chroma's shared ``System`` can stop *after* that check and before
    ``client.get_collection`` runs. The open then raises the stopped-``System``
    error, and the search path reports a backend/palace failure for a palace
    that is readable — NEXA-433. The open must be repaired and retried once.
    """
    from mempalace.backends import PalaceRef, chroma as chroma_module
    from mempalace.backends.chroma import ChromaBackend

    backend = ChromaBackend()
    palace = tmp_path / "palace"
    ref = PalaceRef(id=str(palace), local_path=str(palace))
    try:
        backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)

        client = backend._clients[str(palace)]
        real_get_collection = client.get_collection
        calls = {"n": 0}

        def flaky_get_collection(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise AttributeError("'RustBindingsAPI' object has no attribute 'bindings'")
            return real_get_collection(*args, **kwargs)

        monkeypatch.setattr(client, "get_collection", flaky_get_collection)
        cleared = {"n": 0}
        real_clear = chroma_module._clear_chroma_system_cache

        def counting_clear():
            cleared["n"] += 1
            return real_clear()

        monkeypatch.setattr(chroma_module, "_clear_chroma_system_cache", counting_clear)

        reopened = backend.get_collection(
            palace=ref, collection_name="mempalace_drawers", create=False
        )
    finally:
        backend.close()

    assert calls["n"] == 1, "precondition: the first open used the dead client"
    assert cleared["n"] == 1, "the stopped System must be dropped before the retry"
    assert reopened is not None
