"""A KeyError is not automatically an unknown backend.

``BackendUnavailableError`` — the registry's own ``KeyError`` — is the only
``KeyError`` that means the selected backend *name* is unregistered. Any other
``KeyError`` comes from inside an open backend (a dead chroma handle, a
typed-result key) and used to be reported as ``"Unknown backend"`` with the
palace path in ``details``, sending the reader after the configuration while
the palace was the thing that had failed.
"""

import pytest

from mempalace.backends import BackendUnavailableError


@pytest.fixture
def clean_session_state(monkeypatch):
    """Isolate the mcp_server caches an error path mutates."""
    from mempalace import mcp_server

    for name, value in (
        ("_collection_open_error", None),
        ("_client_cache", None),
        ("_collection_cache", None),
        ("_collection_cache_backend", None),
        ("_collection_cache_palace", None),
        ("_palace_db_inode", 0),
        ("_palace_db_mtime", 0.0),
    ):
        monkeypatch.setattr(mcp_server, name, value)
    return mcp_server


class TestSearcherCollectionOpen:
    def test_registry_keyerror_is_still_an_unknown_backend(self, monkeypatch):
        from mempalace import searcher

        def raise_registry_error(*args, **kwargs):
            raise BackendUnavailableError("bogus", ["chroma", "sqlite_exact"])

        monkeypatch.setattr(searcher, "get_collection", raise_registry_error)

        _col, result = searcher._open_search_collection("/palace", "mempalace_drawers")

        assert result["error"] == "Unknown backend"
        assert "bogus" in result["details"]

    def test_other_keyerror_names_the_backend_not_the_selection(self, monkeypatch):
        from mempalace import searcher

        def raise_dead_handle(*args, **kwargs):
            raise KeyError("/home/jjude/.config/mempalace/palace")

        monkeypatch.setattr(searcher, "get_collection", raise_dead_handle)

        _col, result = searcher._open_search_collection(
            "/home/jjude/.config/mempalace/palace", "mempalace_drawers"
        )

        assert result["error"] == "Backend error"
        assert result["error"] != "Unknown backend"
        # The real cause survives: it is the palace path the lookup used,
        # not a rewrite that hides where the failure came from.
        assert "/home/jjude/.config/mempalace/palace" in result["details"]


class TestVectorDisabledBackendResolution:
    def test_registry_keyerror_is_still_an_unknown_backend(self, monkeypatch):
        from mempalace import searcher

        def raise_registry_error(*args, **kwargs):
            raise BackendUnavailableError("bogus", ["chroma"])

        monkeypatch.setattr(searcher, "resolve_backend_name", raise_registry_error)

        result = searcher._vector_disabled_search(
            query="anything",
            palace_path="/palace",
            wing=None,
            room=None,
            n_results=5,
            collection_name="mempalace_drawers",
        )

        assert result["error"] == "Unknown backend"

    def test_other_keyerror_names_the_backend(self, monkeypatch):
        from mempalace import searcher

        def raise_dead_handle(*args, **kwargs):
            raise KeyError("_type")

        monkeypatch.setattr(searcher, "resolve_backend_name", raise_dead_handle)

        result = searcher._vector_disabled_search(
            query="anything",
            palace_path="/palace",
            wing=None,
            room=None,
            n_results=5,
            collection_name="mempalace_drawers",
        )

        assert result["error"] == "Backend error"
        assert "_type" in result["details"]


class TestSessionCollectionOpen:
    def test_registry_keyerror_is_still_an_unknown_backend(
        self, monkeypatch, config, clean_session_state
    ):
        mcp_server = clean_session_state
        monkeypatch.setattr(mcp_server, "_config", config)

        def raise_registry_error():
            raise BackendUnavailableError("bogus", ["chroma"])

        monkeypatch.setattr(mcp_server, "_selected_backend_name", raise_registry_error)

        assert mcp_server._get_collection(create=False) is None
        assert mcp_server._collection_open_error["error"] == "Unknown backend"

    def test_other_keyerror_names_the_backend(self, monkeypatch, config, clean_session_state):
        mcp_server = clean_session_state
        monkeypatch.setattr(mcp_server, "_config", config)
        monkeypatch.setattr(mcp_server, "_selected_backend_name", lambda: "chroma")

        def raise_dead_handle():
            raise KeyError("_type")

        monkeypatch.setattr(mcp_server, "_get_client", raise_dead_handle)

        assert mcp_server._get_collection(create=True) is None
        error = mcp_server._collection_open_error
        assert error["error"] == "Backend error"
        assert error["error"] != "Unknown backend"
        assert "_type" in error["details"]
