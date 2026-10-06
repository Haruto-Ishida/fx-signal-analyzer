"""「機械学習予測」タブの一部：本番モデルの管理（バージョン・巻き戻し・チャレンジャー）。要件定義書 9-2。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fxtool import data, mlops

from .common import JST, Context


def _jst(s) -> str:
    return "—" if s is None or (isinstance(s, float) and pd.isna(s)) else f"{(pd.Timestamp(s) + JST):%Y/%m/%d %H:%M}"


def render(ctx: Context) -> None:
    con = data.connect()
    try:
        vers = mlops.versions(con, ctx.pair, ctx.tf_key)
        pin = mlops.pinned_version(con, ctx.pair, ctx.tf_key)
        spec = mlops.current_spec(con, ctx.pair, ctx.tf_key)
        ch = mlops.challengers(con, ctx.pair, ctx.tf_key)
        ls = mlops.last_swap(con, ctx.pair, ctx.tf_key)
        cmps = {int(c.id): (mlops.compare_strategy(con, int(c.id), data.now_utc(), ls) if c.kind == "strategy"
                            else mlops.compare(con, int(c.id), data.now_utc(), last_swap=ls))
                for c in ch[ch["status"].isin(["shadow", "ready"])].itertuples()}
    finally:
        con.close()
    ready = ch[ch["status"] == "ready"]
    title = "🗂 本番モデルの管理（バージョン・巻き戻し・チャレンジャー）" + ("　🔔 入れ替えの提案があります" if len(ready) else "")
    with st.expander(title, expanded=bool(len(ready)) or bool(pin)):
        st.markdown(
            f"本番の設定：**{spec.label}**。データが増えただけの再学習は自動で切り替え、毎回バージョンとして保存しています。"
            "設定（特徴量など）を変えるものは「**チャレンジャー**」として、まず**シャドーモード**（裏で判定だけ記録し、シグナルには使わない）で"
            f"{mlops.SHADOW_DAYS}日以上、本番と同じシグナルで成績を比べます。条件を満たしたら提案し、**あなたが承認したときだけ**入れ替えます。"
        )
        # ---------------- バージョン ----------------
        if pin:
            st.warning(f"📌 前のバージョン **{pin}** に戻して固定しています。自動の再学習は止まっています。")
            if st.button("固定を解除する（最新のモデルに戻し、自動の再学習を再開）", key="ml_unpin"):
                con = data.connect()
                try:
                    mlops.unpin(con, ctx.pair, ctx.tf_key)
                finally:
                    con.close()
                st.rerun()
        if vers.empty:
            st.caption("まだ本番のモデルがありません。")
        else:
            active = pin or vers["version"].iat[0]
            st.markdown("**バージョンの履歴**（新しい順・最大10件）")
            st.dataframe(pd.DataFrame({
                "バージョン": vers["version"].head(10),
                "学習した日時": vers["trained_at"].head(10).map(_jst),
                "件数": vers["n_train"].head(10),
                "特徴量": vers["features"].head(10).fillna("v2").map({"v2": "これまで", "v2+fd": "分数階差分あり"}),
                "条件": vers["ok"].head(10).map({1: "満たす", 0: "参考値"}),
                "使用中": (vers["version"].head(10) == active).map({True: "✅", False: ""}),
            }), hide_index=True, width="stretch")
            older = [v for v in vers["version"].head(10) if v != active]
            if older:
                c1, c2 = st.columns([3, 1])
                target = c1.selectbox("前のバージョンに戻す", older, key="ml_rollback_target")
                if c2.button("このバージョンに戻す", key="ml_rollback"):
                    con = data.connect()
                    try:
                        mlops.rollback(con, ctx.pair, ctx.tf_key, target, data.now_utc())
                    finally:
                        con.close()
                    st.rerun()

        # ---------------- チャレンジャー ----------------
        st.markdown("**チャレンジャー**")
        if ch.empty:
            st.caption("まだありません。")
        for c in ch.itertuples():
            sp = mlops.SPECS.get(c.spec)
            name = ("機械学習：" + sp.label) if sp else mlops.strategy_label(c)
            status = {"shadow": "シャドーモード中", "ready": "🔔 入れ替えの提案中", "approved": "承認して入れ替え済み",
                      "rejected": "見送り"}.get(c.status, c.status)
            st.markdown(f"- **{name}**（{status}・{_jst(c.created_at)} から）")
            cmp = cmps.get(int(c.id))
            if cmp is not None:
                st.dataframe(pd.DataFrame([{"条件": n, "結果": "✅" if ok else "—", "内容": d} for n, ok, d in cmp.checks]),
                             hide_index=True, width="stretch")
                if c.kind == "strategy":
                    st.caption(f"シャドーモードで決済したシグナル {cmp.n} 件の損益（コスト後）：{cmp.pips:+,.1f} pips。"
                               "本番には使わず、出ていたはずのシグナルを記録しているだけです。")
                else:
                    st.caption(f"同じシグナル {cmp.n} 件での損益：本番 {cmp.pips_champ:+,.1f} pips ／ チャレンジャー {cmp.pips_chal:+,.1f} pips")
            if c.status == "ready":
                b1, b2 = st.columns(2)
                if b1.button("✅ 承認して入れ替える", key=f"approve_{c.id}"):
                    con = data.connect()
                    try:
                        mlops.approve(con, ctx.pair, ctx.tf_key, int(c.id), data.now_utc())
                    finally:
                        con.close()
                    st.rerun()
                if b2.button("見送る", key=f"reject_{c.id}"):
                    con = data.connect()
                    try:
                        mlops.reject(con, int(c.id), data.now_utc())
                    finally:
                        con.close()
                    st.rerun()
        active_specs = set(ch[ch["status"].isin(["shadow", "ready"])]["spec"]) if len(ch) else set()
        options = [k for k, s in mlops.SPECS.items() if s.features != spec.features and k not in active_specs]
        if options and not vers.empty:
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("新しいチャレンジャー", options, format_func=lambda k: f"{mlops.SPECS[k].label}：{mlops.SPECS[k].description}",
                                key="ml_new_challenger")
            if c2.button("シャドーモードで始める", key="ml_register",
                         help="本番には影響しません。試した組み合わせが1つ増えます。"):
                con = data.connect()
                try:
                    mlops.register(con, ctx.pair, ctx.tf_key, pick, data.now_utc())
                finally:
                    con.close()
                st.rerun()
