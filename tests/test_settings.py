from fxtool import settings


def test_project_settings_file_loads_without_warnings():
    s = settings.load()
    assert s.warnings == ()
    assert s.paper.initial_balance == 100_000
    assert s.kill_switch.max_consecutive_losses == 5 and s.kill_switch.max_signals_per_day == 10
    assert s.kill_switch.daily_loss_pct == 3.0 and s.kill_switch.weekly_loss_pct == 6.0
    assert s.trial_budget.ga_full_search == 500 and s.trial_budget.ga_weekly_search == 100


def test_missing_values_use_defaults_and_bad_values_are_reported(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[paper]\ninitial_balance = "たくさん"\n[sizing]\nkelly_fraction = 1.0\nmax_risk_pct = 5\n',
                 encoding="utf-8")
    s = settings.load(p)
    assert s.paper.initial_balance == 100_000           # 数値でない → 初期値
    assert s.sizing.kelly_fraction == 0.5               # 上限 0.5 に抑える
    assert s.sizing.max_risk_pct == 2.0                 # 上限 2% に抑える
    assert len(s.warnings) == 3


def test_broken_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text("[paper\ninitial_balance = ", encoding="utf-8")
    s = settings.load(p)
    assert s.paper == settings.Paper() and s.warnings
