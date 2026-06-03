"""Tests for healthcheck and metrics endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from unittest.mock import patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with localhost trust for auth."""
    with patch("backend.config.VOCALIE_TRUST_LOCALHOST", True):
        from backend.app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"ok", "degraded"}
        assert "api_version" in data
        assert "uptime_s" in data
        assert "timestamp" in data
        assert "work_dir_writable" in data
        assert "output_dir_writable" in data

    def test_health_includes_backends(self, client):
        response = client.get("/v1/health")
        data = response.json()
        # backends may be None if probing fails, but should be present key
        assert "backends" in data

    def test_health_degraded_when_dir_not_writable(self, client):
        with patch("backend.routes.health._check_dir_writable", return_value=False):
            response = client.get("/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"


class TestMetricsEndpoint:
    def test_metrics_returns_data(self, client):
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_s" in data
        assert "jobs_total" in data
        assert "jobs_completed" in data
        assert "jobs_failed" in data
        assert "jobs_pending" in data
        assert "backends_available" in data
        assert "work_dir_writable" in data
        assert "output_dir_writable" in data

    def test_metrics_initial_state(self, client):
        response = client.get("/v1/metrics")
        data = response.json()
        # Metrics may accumulate across tests in shared process; verify types and non-negative
        assert isinstance(data["jobs_total"], int)
        assert data["jobs_total"] >= 0
        assert isinstance(data["jobs_completed"], int)
        assert data["jobs_completed"] >= 0
        assert isinstance(data["jobs_failed"], int)
        assert data["jobs_failed"] >= 0
        assert isinstance(data["jobs_pending"], int)
        assert data["jobs_pending"] >= 0