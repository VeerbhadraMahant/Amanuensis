"""System-wide text injection (F8): types committed text into whichever window has focus.

Windows backend only (SendInput with Unicode key events, so any script works regardless of keyboard layout).
Never enabled implicitly: live.py needs --inject, because this writes into other applications.
"""
import sys
from collections.abc import Callable

from amanuensis.asr.streaming import Update


def _utf16_units(text: str) -> list[int]:
    data = text.encode("utf-16-le")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]


def _send_input_windows(text: str) -> None:
    import ctypes
    from ctypes import wintypes

    KEYEVENTF_UNICODE, KEYEVENTF_KEYUP, INPUT_KEYBOARD = 0x0004, 0x0002, 1

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class MOUSEINPUT(ctypes.Structure):  # present so the union has the size SendInput expects
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    events = []
    for unit in _utf16_units(text):  # surrogate pairs go out as two units, which is what Windows expects
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            events.append(INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(0, unit, flags, 0, 0))))
    array = (INPUT * len(events))(*events)
    sent = ctypes.windll.user32.SendInput(len(events), array, ctypes.sizeof(INPUT))
    if sent != len(events):
        raise OSError(f"SendInput delivered {sent} of {len(events)} events (blocked by a higher-privilege window?)")


def default_backend() -> Callable[[str], None]:
    if sys.platform == "win32":
        return _send_input_windows
    raise NotImplementedError("text injection is only implemented for Windows (add an adapter for your OS)")


class InjectionSink:
    """Update sink: types each newly committed chunk plus a space. Tentative text is never injected,
    because it can still change and typed characters cannot be taken back."""

    def __init__(self, backend: Callable[[str], None] | None = None) -> None:
        self._type = backend or default_backend()

    def __call__(self, update: Update) -> None:
        if update.committed:
            self._type(update.committed + " ")
