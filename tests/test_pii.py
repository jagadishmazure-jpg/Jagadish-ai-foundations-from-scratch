import pytest

from aifound.pii import SafeError, has_pii, scrub


@pytest.mark.parametrize(
    "text,tag",
    [("card 4111 1111 1111 1111", "[CARD]"), ("ssn 123-45-6789", "[SSN]"), ("a@b.co", "[EMAIL]"), ("call (555) 123-4567", "[PHONE]")],
)
def test_scrub_masks(text, tag):
    assert tag in scrub(text)


def test_scrub_leaves_order_numbers():
    assert not has_pii("order 1234567890123")  # fails Luhn


def test_invalid_ssn_area_not_masked():
    assert scrub("000-12-3456") == "000-12-3456"


def test_safe_error_scrubs_message():
    assert "x@y.com" not in str(SafeError("bad x@y.com"))
