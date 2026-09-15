"""Checks for the consolidated public experiment commands."""

from types import SimpleNamespace

from closeloop_experiments import cli


def test_campaign_rejects_analyzer_actions(capsys):
    """Offline analysis is available only through closeloop_analyzer."""
    assert cli.campaign_main([
        "input-variation", "analyze", "study.yaml",
        "--artifact-root", "/tmp/artifacts",
    ]) == 2
    assert "unsupported" in capsys.readouterr().err


def test_campaign_forwards_explicit_artifact_root(monkeypatch):
    """Named campaign dispatch never invents an artifact location."""
    received = []
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            main=lambda arguments: received.extend(arguments) or 0
        ),
    )
    assert cli.campaign_main([
        "communication", "generate", "study.yaml",
        "--artifact-root", "/tmp/artifacts",
    ]) == 0
    assert received == [
        "--artifact-root", "/tmp/artifacts", "generate", "study.yaml",
    ]
