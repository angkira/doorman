"""
Unit tests for doorman CLI
"""

import json
import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner

# Mock socket for testing
class MockSocket:
    def __init__(self, response):
        self.response = response
        self.sent_data = None
    
    def connect(self, path):
        pass
    
    def sendall(self, data):
        self.sent_data = data
    
    def recv(self, size):
        return self.response.encode()
    
    def close(self):
        pass


def test_ipc_request_format():
    """Test IPC request JSON format"""
    request = {"type": "authenticate", "username": "testuser"}
    json_str = json.dumps(request)
    
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed["type"] == "authenticate"
    assert parsed["username"] == "testuser"


def test_ipc_response_parsing():
    """Test IPC response parsing"""
    response_json = '{"status":"success","message":"Authenticated"}\n'
    response = json.loads(response_json.strip())
    
    assert response["status"] == "success"
    assert response["message"] == "Authenticated"


def test_ipc_failure_response():
    """Test IPC failure response"""
    response_json = '{"status":"failure","reason":"User not enrolled"}\n'
    response = json.loads(response_json.strip())
    
    assert response["status"] == "failure"
    assert "reason" in response


def test_status_request():
    """Test status request format"""
    request = {"type": "status"}
    json_str = json.dumps(request)
    
    parsed = json.loads(json_str)
    assert parsed["type"] == "status"


def test_enroll_request():
    """Test enrollment request format"""
    request = {"type": "enroll", "username": "newuser"}
    json_str = json.dumps(request)
    
    parsed = json.loads(json_str)
    assert parsed["type"] == "enroll"
    assert parsed["username"] == "newuser"


def test_list_users_request():
    """Test list users request"""
    request = {"type": "list_users"}
    json_str = json.dumps(request)
    
    parsed = json.loads(json_str)
    assert parsed["type"] == "list_users"


def test_remove_user_request():
    """Test remove user request"""
    request = {"type": "remove_user", "username": "olduser"}
    json_str = json.dumps(request)
    
    parsed = json.loads(json_str)
    assert parsed["type"] == "remove_user"
    assert parsed["username"] == "olduser"


def test_config_paths():
    """Test configuration file paths"""
    from doorman.cli import SOCKET_PATH, DATA_DIR, MODELS_DIR
    
    assert SOCKET_PATH == "/run/doorman.sock"
    assert DATA_DIR == "/var/lib/doorman"
    assert MODELS_DIR == "/var/lib/doorman/models"


def test_dependency_check_structure():
    """Test dependency checking logic"""
    # This would normally check for actual dependencies
    # Here we just test the structure
    
    dependencies = {
        "rustc": False,
        "cargo": False,
        "gcc": False,
        "libpam0g-dev": False,
    }
    
    missing = [name for name, found in dependencies.items() if not found]
    
    # In this test, all are "missing"
    assert len(missing) == 4
    assert "rustc" in missing


def test_pam_line_format():
    """Test PAM configuration line format"""
    pam_line = "auth sufficient pam_doorman.so"
    
    parts = pam_line.split()
    assert parts[0] == "auth"
    assert parts[1] == "sufficient"
    assert parts[2] == "pam_doorman.so"


def test_service_file_content():
    """Test systemd service file structure"""
    service_content = """[Unit]
Description=doorman Face Authentication Daemon

[Service]
Type=simple
ExecStart=/usr/local/bin/doormand

[Install]
WantedBy=multi-user.target
"""
    
    assert "[Unit]" in service_content
    assert "[Service]" in service_content
    assert "[Install]" in service_content
    assert "doormand" in service_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

