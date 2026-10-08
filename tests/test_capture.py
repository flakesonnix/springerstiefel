from springerstiefel import capture


def test_matches_chat_api_requests():
    assert capture.is_interesting_url("https://hey.bild.de/api/chat")
    assert capture.is_interesting_url(
        "https://example.com/graphql?op=SendMessage"
    )
    assert capture.is_interesting_url(
        "https://example.com/v1/completions"
    )


def test_ignores_static_assets():
    assert not capture.is_interesting_url(
        "https://hey.bild.de/_next/static/chunk.js"
    )
    assert not capture.is_interesting_url("https://hey.bild.de/favicon.ico")


def test_matching_is_case_insensitive():
    assert capture.is_interesting_url("https://example.com/API/CHAT")
