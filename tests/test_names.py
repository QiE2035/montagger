"""Name normalization parity with monbooru's tags.NormalizeName."""

from montagger.engine.names import has_tag_content, normalize_name, sanitize_label


def test_basic_folding():
    assert normalize_name("Hatsune Miku") == "hatsune_miku"
    assert normalize_name("  1GIRL  ") == "1girl"
    assert normalize_name("multiple   spaces") == "multiple_spaces"


def test_reserved_characters_fold():
    assert normalize_name('foo"bar') == "foo_bar"
    assert normalize_name("a*b") == "a_b"


def test_leading_underscore_suppressed():
    assert normalize_name("  leading") == "leading"
    assert normalize_name("\t\tx") == "x"


def test_decoration_only_is_empty():
    assert normalize_name(' "_" ') == ""
    # Parens survive normalization; the placeholder decision happens in
    # sanitize_label, exactly as the Go original layers it.
    assert normalize_name("()") == "()"
    name, ok = sanitize_label("()", 3)
    assert (name, ok) == ("_unsupported_3", False)


def test_has_tag_content():
    assert has_tag_content("1girl")
    assert has_tag_content(":-d")  # a real emoticon tag: 'd' is not decoration
    assert not has_tag_content("_()")
    assert not has_tag_content(":-)")  # every char is decoration per Go's class


def test_sanitize_placeholder():
    name, ok = sanitize_label("", 7)
    assert name == "_unsupported_7" and not ok
    name, ok = sanitize_label("Valid Tag", 0)
    assert name == "valid_tag" and ok


def test_long_name_truncated():
    name, ok = sanitize_label("x" * 500, 0)
    assert ok and len(name) == 200
