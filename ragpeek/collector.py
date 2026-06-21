from __future__ import annotations
from contextvars import ContextVar, Token
from typing import Optional
from ragpeek.session import TraceSession

# the ContextVar — stores the session_id of the currently active trace.
# ContextVar propagates correctly through async code and is safe with threading.
_current_session_id: ContextVar[Optional[str]] = ContextVar(
    "_current_session_id", default=None
)


class TraceCollector:
    """
    Singleton. Maintains the registry of active TraceSession objects.
    All parts of the system interact with it via module-level helpers.
    """

    def __init__(self):
        self._sessions: dict[str, TraceSession] = {}
        self._session_tokens: dict[str, Token[Optional[str]]] = {}

    def start_session(self, query: str) -> TraceSession:
        session = TraceSession(query=query)
        self._sessions[session.session_id] = session
        self._session_tokens[session.session_id] = _current_session_id.set(
            session.session_id
        )
        return session

    def get_current_session(self) -> Optional[TraceSession]:
        session_id = _current_session_id.get()
        if session_id is None:
            return None
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[TraceSession]:
        session = self._sessions.pop(session_id, None)
        token = self._session_tokens.pop(session_id, None)
        if session:
            session.finalise()
        if token is not None:
            _current_session_id.reset(token)
        else:
            _current_session_id.set(None)
        return session

    def clear(self) -> None:
        """Used in tests to reset state between runs."""
        self._sessions.clear()
        self._session_tokens.clear()
        _current_session_id.set(None)


# module-level singleton — import this everywhere
_collector = TraceCollector()


def get_collector() -> TraceCollector:
    return _collector
