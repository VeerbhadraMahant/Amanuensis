from amanuensis.asr.streaming import Update
from amanuensis.output.inject import InjectionSink, _utf16_units


def test_only_committed_text_is_typed_with_trailing_space():
    typed = []
    sink = InjectionSink(backend=typed.append)
    sink(Update("hello world", "tentative words", False))
    sink(Update("", "still tentative", False))  # nothing committed: nothing typed
    sink(Update("again", "", True))
    assert typed == ["hello world ", "again "]


def test_unicode_units_include_surrogate_pairs():
    assert _utf16_units("aé") == [0x61, 0xE9]
    assert _utf16_units("\U0001F600") == [0xD83D, 0xDE00]  # outside the BMP: two units
