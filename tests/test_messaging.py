from agents.Messaging.messaging_agent import parse_analysis_response, clean_ai_text


def test_parse_analysis_response_with_labels():
    raw = "sentiment: positive\nintent: ask for project status and next steps"
    result = parse_analysis_response(raw)

    assert result["sentiment"] == "positive"
    assert result["intent"] == "ask for project status and next steps"


def test_clean_ai_text_strips_labels_and_whitespace():
    raw = "Assistant:  Thanks for the update.\n\nI will review the next deliverable."
    cleaned = clean_ai_text(raw)

    assert cleaned == "Thanks for the update.\n\nI will review the next deliverable."
