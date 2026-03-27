"""Behavior-focused tests for DeepResearch FastAPI backend."""


class TestHealth:
    def test_health_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


class TestStateManagement:
    def test_setup_resets_state(self, client):
        # Do some operations first
        client.post("/search", json={"query": "python programming"})
        client.post("/answer", json={"final_answer": "test"})

        # Reset
        client.post("/setup")

        # Verify state is cleared
        state = client.get("/state").json()
        assert state["search_count"] == 0
        assert state["fetch_count"] == 0
        assert state["submitted_answer"] is None

    def test_state_tracks_counts(self, client):
        # Initial state
        state = client.get("/state").json()
        assert state["search_count"] == 0
        assert state["fetch_count"] == 0

        # Search increments count
        client.post("/search", json={"query": "python programming"})
        state = client.get("/state").json()
        assert state["search_count"] == 1

        # Fetch increments count
        client.post("/fetch", json={"url": "https://example.com"})
        state = client.get("/state").json()
        assert state["fetch_count"] == 1


class TestSearch:
    def test_search_returns_results_with_title_and_url(self, client):
        resp = client.post("/search", json={"query": "python programming language"})
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) > 0
        # Either a result with title/url, or a "no results" message
        first = results[0]
        assert "title" in first or "message" in first


class TestFetch:
    def test_fetch_returns_content(self, client):
        resp = client.post("/fetch", json={"url": "https://example.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert isinstance(data["content"], str)

    def test_fetch_invalid_url_returns_400(self, client):
        resp = client.post("/fetch", json={"url": "not-a-valid-url"})
        assert resp.status_code == 400


class TestAnswer:
    def test_answer_stores_and_evaluate_correct(self, client):
        # Submit answer
        client.post("/answer", json={"final_answer": "The answer is Paris"})

        # Verify stored
        state = client.get("/state").json()
        assert state["submitted_answer"] == "The answer is Paris"

        # Evaluate with matching expected (case-insensitive substring)
        resp = client.post("/evaluate", json={"expected_answer": "paris"})
        data = resp.json()
        assert data["reward"] == 1.0
        assert data["done"] is True

    def test_evaluate_incorrect_returns_zero(self, client):
        client.post("/answer", json={"final_answer": "London"})
        resp = client.post("/evaluate", json={"expected_answer": "paris"})
        assert resp.json()["reward"] == 0.0

    def test_evaluate_no_submission_returns_zero(self, client):
        # Don't submit any answer, just evaluate
        resp = client.post("/evaluate", json={"expected_answer": "paris"})
        data = resp.json()
        assert data["reward"] == 0.0
        assert data["done"] is False
