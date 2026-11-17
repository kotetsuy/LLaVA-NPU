import base64

from src.inference.vlm_worker import VlmServerWorker


def test_frames_are_sent_in_order(monkeypatch):
    sent = {}

    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "動いています。"}}]}

    def post(url, json, timeout):
        sent.update(json)
        return Response()

    monkeypatch.setattr("src.inference.vlm_worker.requests.post", post)
    result = VlmServerWorker().predict_frames([b"first", b"second"])
    content = sent["messages"][-1]["content"]
    urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
    assert [base64.b64decode(url.split(",", 1)[1]) for url in urls] == [b"first", b"second"]
    assert result.caption == "動いています。"
