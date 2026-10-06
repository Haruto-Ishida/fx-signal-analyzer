from fxtool import settings


def test_project_settings_file_loads_without_warnings():
    s = settings.load()
    assert s.warnings == ()
    assert s.paper.initial_balance == 100_000
    assert s.kill_switch.max_consecutive_losses == 5 and s.kill_switch.max_signals_per_day == 10
    assert s.kill_switch.daily_loss_pct == 3.0 and s.kill_switch.weekly_loss_pct == 6.0
    assert s.trial_budget.ga_full_search == 500 and s.trial_budget.ga_weekly_search == 150
    # 2026-10-07：GA の探索をやめて全通りの評価に切り替えたので、定期的な再探索は止めている
    assert s.trial_budget.ga_weekly_enabled is False and s.trial_budget.ga_interval_weeks == 3
    assert s.review.interval_months == 12 and s.review.notice_days == 30
    # 2026-10-07 に計算した、通貨ペアごとの1日のシグナルの回数の上限
    assert s.signal_caps == {"USD/JPY": 13, "EUR/JPY": 13, "GBP/JPY": 13, "AUD/JPY": 13,
                             "EUR/USD": 14, "GBP/USD": 14, "AUD/USD": 14}
    assert s.signal_cap("USD/JPY") == 13 and s.signal_cap("XXX/YYY") == 10


def test_signal_caps_validation(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[signal_caps]\n"USD/JPY" = 12\n"EUR/USD" = "たくさん"\n"GBP/USD" = 0\n', encoding="utf-8")
    s = settings.load(p)
    assert s.signal_caps == {"USD/JPY": 12} and len(s.warnings) == 2


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
