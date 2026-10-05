# FX予測ツール 研究ノート（実装の指針）

このファイルは、FX売買シグナル分析ツールを作るうえで参考にする研究の要点と、
そこから導いた実装ルールをまとめたものです。コードを書く前に必ず読み、
ここに書かれた原則に反する実装をする場合は、理由を説明してから行うこと。

---

## 1. 前提として押さえる研究上の事実

### 1-1. 為替の予測はとても難しい
- **Meese & Rogoff（1983, Journal of International Economics）**
  経済理論に基づく為替モデルが、短期の予測では「明日も今日と同じ」という
  単純なランダムウォーク予想に勝てなかったことを示した古典的研究。
  → 実装ルール：どんなモデルも必ず「ランダムウォーク（前日と同じ）」
    「ランダム売買」「何もしない」と比較する。これに勝てないモデルは採用しない。

### 1-2. 効いた手法も時間とともに効かなくなる
- **Fischer & Krauss（2018, European Journal of Operational Research）**
  LSTMでS&P500構成銘柄の翌日の上げ下げを予測。取引コスト前では
  ランダムフォレスト等を上回ったが、2010年以降は超過収益がほぼ消え、
  コスト後はゼロ前後になった。
  → 実装ルール：成績は期間ごとに分けて表示し、直近の成績の劣化を監視する。
    全期間の平均だけで判断しない。

### 1-3. テクニカル分析の有効性は「あるが限定的」
- **Menkhoff & Taylor（2007, Journal of Economic Literature）**
  FXのプロは特に短期の判断でテクニカル分析を広く使っている。
  ただし利益が出るかは時期や通貨によってまちまちで、時代とともに弱まる傾向。
- **Neely, Weller & Dittmar（1997, Journal of Financial and Quantitative Analysis）**
  遺伝的プログラミングで FX の売買ルールを自動探索し、
  一部の通貨ペアで検証期間でも超過収益が見られた。「自動で戦略を探す」先駆的研究。
- **Park & Irwin（2007, Journal of Economic Surveys）**
  テクニカル分析の収益性を調べた研究の多くは利益を報告しているが、
  データの使い回し（データスヌーピング）、取引コストの扱い、
  リスク調整の不足など、検証方法に問題があるものが多いと指摘。
  → 実装ルール：戦略の自動探索をするなら、試した戦略の数をすべて記録し、
    多重検定の補正（2-3参照）を行う。

---

## 2. 正しい検証方法（最重要）

主な出典：**Marcos López de Prado『Advances in Financial Machine Learning』（2018, Wiley）**

### 2-1. 未来の情報を混ぜない（リーク防止）
- 特徴量は、その時点で実際に知り得たデータだけで計算する。
  例：その日の終値で計算した指標を、その日の始値での売買判断に使わない。
- 標準化・正規化のパラメータ（平均や標準偏差）は学習データだけで計算する。
  全期間のデータで計算してから分割するのは禁止。
- 欠損値の補完で後ろのデータを使わない（前方補完のみ）。

### 2-2. 時系列のための交差検証
- 普通のランダムなK分割交差検証は使わない（未来のデータで学習してしまう）。
- **Purged K-Fold**：学習データと検証データの境目で、ラベル期間が重なる
  サンプルを学習データから取り除く（パージ）。
- **Embargo（エンバーゴ）**：検証期間の直後の一定期間も学習から外す。
- **ウォークフォワード**：「過去N期間で学習 → 次のM期間で検証」を
  少しずつずらしながら繰り返す。本番に最も近い検証方法。
- **CPCV（Combinatorial Purged Cross-Validation）**：複数の検証経路を作り、
  成績のばらつきを見る。余裕があれば実装する。

### 2-3. バックテストの過学習を疑う
- **Bailey, Borwein, López de Prado & Zhu「The Probability of Backtest Overfitting」
  （Journal of Computational Finance, 2017）**
  たくさんの戦略を試して一番良いものを選ぶと、その好成績は偶然である
  可能性が高い。その確率（PBO）を CSCV という方法で推定できる。
- **Bailey & López de Prado「The Deflated Sharpe Ratio」
  （Journal of Portfolio Management, 2014）**
  試した戦略の数やリターン分布の歪みを考慮して、シャープレシオを割り引いて評価する。
- → 実装ルール：
  - 試した戦略・パラメータの組み合わせの数を必ず記録する。
  - 最良戦略について PBO と Deflated Sharpe Ratio を計算して表示する。
  - 最後の一定期間（例：直近20%）は「最終テスト用」として開発中は一切使わない。

---

## 3. ラベル（正解データ）の作り方

出典：López de Prado（2018）

- **トリプルバリア法**：エントリー後に「利確ライン」「損切りライン」
  「時間切れ」の3つのうち、どれに最初に到達したかでラベルを付ける。
  単純な「翌日上がったか下がったか」より実際の売買に近い。
  ライン幅は ATR などのボラティリティに比例させる。
- **メタラベリング**：まずテクニカル指標などで売買の方向を決め、
  機械学習モデルには「そのシグナルに乗るべきか見送るべきか」だけを学習させる。
  初心者向けの構成として最もおすすめ。
- **サンプルの重複**：ラベル期間が重なるサンプルは独立ではないので、
  重複度に応じてサンプルの重みを下げる。
- **分数階差分（Fractional Differentiation）**：価格の「記憶」を残しつつ
  定常化する手法。余裕があれば特徴量に加える。

---

## 4. モデルの選び方

### 4-1. まずはシンプルなモデルから
- 順番：ロジスティック回帰 → ランダムフォレスト → 勾配ブースティング（LightGBM等）
  → LSTM/GRU。
- 複雑なモデルは、シンプルなモデルにウォークフォワード検証で
  コスト後の成績で勝ったときだけ採用する。

### 4-2. 参考になる研究
- **Gu, Kelly & Xiu「Empirical Asset Pricing via Machine Learning」
  （Review of Financial Studies, 2020）**
  株式リターン予測で多くの機械学習手法を比較。決定木系やニューラルネットが
  線形モデルを上回り、モメンタム・流動性・ボラティリティ系の変数が重要だった。
- **Fischer & Krauss（2018）**：LSTM の具体的な構成例（1-2参照）。
- **Hu ほか「A Survey of Forex and Stock Price Prediction Using Deep Learning」
  （arXiv:2103.09750, 2021）**
  CNN、LSTM、DNN、RNN、強化学習などで FX・株を予測した研究を手法別に整理。
- **Islam ほか「A Review on Recent Advancements in FOREX Currency Prediction」
  （Algorithms, 2020, 13(8):186）**
  2017〜2019年の FX 予測研究39本のレビュー。LSTM や GRU が有望とされる。
- **Sezer, Gudelek & Ozbayoglu「Financial time series forecasting with deep learning:
  A systematic literature review: 2005–2019」（Applied Soft Computing, 2020）**
  金融時系列のディープラーニング研究の総まとめ。

### 4-3. 強化学習について
- レビュー論文では高い成績の報告もあるが、過学習しやすく検証が難しい。
  上記の手法で土台ができてから、実験として扱う。

---

## 5. 評価指標

最近の FX 予測研究は統計的な精度ばかり報告し、実際の損益での評価が
不足していると指摘されている。このツールでは両方を必ず出す。

- **損益系（優先）**：取引コスト（スプレッド）控除後の損益、シャープレシオ、
  最大ドローダウン、プロフィットファクター、勝率、平均利益/平均損失、取引回数
- **統計系**：方向的中率、確率予測の較正（予測確率60%のとき実際に60%当たるか）
- 取引回数が少ない戦略（例：100回未満）の結果は「参考値」として扱う。

---

## 6. リスク管理（将来の自動売買に向けて）

- ポジションサイズは口座残高の一定割合（例：1回の損失が残高の1〜2%以内）、
  ATR に応じて調整する。
- 1日・1週間の最大損失に達したら、その期間は新規シグナルを出さない。
- 重要な経済指標の発表前後や、週明けの窓開け、スプレッドが広がる時間帯は
  シグナルを控えるか注意表示を出す。
- 実際の注文機能は作らない。作る場合も、デモ口座で長期間検証してから。

---

## 7. よくある落とし穴チェックリスト

実装後、以下を自分でチェックして結果を報告すること。

- [ ] 未来のデータが特徴量や正規化に混ざっていないか
- [ ] 取引コストを含めているか
- [ ] ランダム売買・ランダムウォークと比較したか
- [ ] 最終テスト期間を開発中に使っていないか
- [ ] 試した戦略の数を記録し、過学習の補正をしたか
- [ ] 期間別の成績を見て、直近で劣化していないか確認したか
- [ ] タイムゾーンと週末のデータの扱いは正しいか
- [ ] 成績が良すぎる場合、まずバグやリークを疑ったか

---

## 参考文献一覧

1. Meese, R. & Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*.
2. Neely, C., Weller, P. & Dittmar, R. (1997). Is Technical Analysis in the Foreign Exchange Market Profitable? A Genetic Programming Approach. *Journal of Financial and Quantitative Analysis*.
3. Menkhoff, L. & Taylor, M. (2007). The Obstinate Passion of Foreign Exchange Professionals: Technical Analysis. *Journal of Economic Literature*.
4. Park, C.-H. & Irwin, S. (2007). What Do We Know About the Profitability of Technical Analysis? *Journal of Economic Surveys*.
5. Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
6. Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
7. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
8. Fischer, T. & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.
9. Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*.
10. Islam, M. S. et al. (2020). A Review on Recent Advancements in FOREX Currency Prediction. *Algorithms*, 13(8), 186.
11. Sezer, O. B., Gudelek, M. U. & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. *Applied Soft Computing*.
12. Hu, Z. et al. (2021). A Survey of Forex and Stock Price Prediction Using Deep Learning. arXiv:2103.09750.
