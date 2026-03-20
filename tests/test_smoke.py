from enterprise_rag.main import main


def test_main_runs(capsys):
    main([])
    captured = capsys.readouterr()
    assert "Query:" in captured.out
    assert "IdentityHub Enterprise SSO" in captured.out
