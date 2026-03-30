from threading import Lock

from app.models import MessageEvent


class MessageBus:
    def __init__(self) -> None:
        self._events: list[MessageEvent] = []
        self._lock = Lock()

    def publish(
        self,
        topic: str,
        sender: str,
        payload: dict,
        recipient: str | None = None,
    ) -> MessageEvent:
        event = MessageEvent(topic=topic, sender=sender, recipient=recipient, payload=payload)
        with self._lock:
            self._events.append(event)
        return event

    def history(self) -> list[MessageEvent]:
        with self._lock:
            return list(self._events)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)
