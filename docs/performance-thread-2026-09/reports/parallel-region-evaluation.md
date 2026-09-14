# 領域判定の追加並列化

比較元は `49167ba114db3100f369547e5b0c361644b1627f`。
四色定理の記事を契機に、独立した判定をまとめる方法を検証した。
四色彩色そのものは導入していない。

## 実装

- `gradient::merge_partition` の統合後の近隣候補を、変更されない領域状態に対して並列評価する。領域のペアを従来どおり正規化し、近隣領域番号順に優先度キューへ戻す。同点時の連番と貪欲統合の採用順を維持する。
- 微小領域の吸収判定を `micro_region_proposal` に分離する。各判定は同じ領域状態を参照し、変更画素と材料統合の判定を返す。結果を領域番号順に集めた後、既存の連鎖解決・画素更新を実行する。
- 近傍の画素収集用配列は処理チャンクごとに確保・再利用する。チャンク数をワーカー数の4倍以下に抑え、領域ごとの全領域サイズ配列の確保を避ける。既存のRayonプールを使う。

領域の統合条件、局所色とアルファの証拠、曲線近似、SVG描画順、RGBAと拡大描画の検証条件は変更しない。隣接しない領域同士でも統合先を共有し得るため、統合の実行自体は逐次のままとした。

## 比較方法

変更前後を `cargo build --release --locked --features diagnostics` でビルドし、別名で保存する。既存の `scripts/benchmark_exact.py` で交互に実行し、SVGのSHA-256と実行時間・出力先を除く全診断値を比較する。速度測定はビルド・テスト終了後に行う。

スレッド数による判定の変化は、ワイパーと報告済みの微小領域2例を使った追加テストで確認する。1・2・4ワーカー間で、統合数、領域ラベル、塗りの正規化画素、推定に残す画素、再割り当て画素数を比較する。

## 測定結果

AMD Ryzen AI 9 465 / x86_64、Rust 1.95.0。時間は起動と入出力を含む実測秒数。以下は4スレッド・各3回の中央値。

| ケース | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|
| car | 39.931 | 39.092 | 2.1% |
| wiki | 27.861 | 27.267 | 2.1% |
| boy | 6.879 | 6.821 | 0.8% |
| alpha | 0.899 | 0.899 | 0.0% |
| key | 20.275 | 18.972 | 6.4% |

全30回で、各ケースのSVGのSHA-256と実行時間・出力先を除く全診断値が一致した。同一SVGなので透過・拡大時の描画も維持される。

変更対象を含む段階の時間は次のとおり。`segmentation` は微小領域の吸収だけでなく、従来の領域分割も含む。`paint-aware-merge` は初期候補評価や統合の実行も含む。

| ケース | segmentation 前→後（秒） | paint-aware-merge 前→後（秒） |
|---|---:|---:|
| car | 1.399 → 0.995 | 1.375 → 0.848 |
| wiki | 1.355 → 0.904 | 0.817 → 0.495 |
| key | 0.469 → 0.331 | 0.695 → 0.328 |

これらのケースでは領域分割段階が約29–33%、塗り統合段階が約38–53%短縮した。ただし、全体時間には変更していない段階の変動も含まれる。車の2回目は全体時間が逆転した。3回の中央値はこの環境での観測値であり、小さな差を一般的な速度保証とはしない。イラストと透明画像の改善は限定的だった。

### 追加経路（各1回の参考値）

| ケース | スレッド数 | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|---:|
| car | 1 | 53.459 | 53.701 | -0.5% |
| alpha | 1 | 1.368 | 1.363 | 0.3% |
| photo-small | 4 | 21.994 | 19.759 | 10.2% |
| adaptive-small | 4 | 46.945 | 47.808 | -1.8% |
| car-3passes | 4 | 43.393 | 38.301 | 11.7% |

負の短縮率は遅くなったことを示す。単発のため、ばらつきと変更の影響を分離した値ではない。1スレッドの車と高精細化では時間が増加しており、すべてのケースで高速化するとは主張しない。

`photo-small` は640×426の viewport1、`adaptive-small` は cliparts の初期最大寸法512、`car-3passes` は塗り統合3パス。高精細化は両版とも2候補を変換・評価し、品質判定で2候補とも棄却した。この測定は子変換と棄却経路を確認したもので、採用される高精細化や巨大画像全体の完走性能を実証するものではない。

追加分も含む全40回で、各ケース・スレッド数における変更前後のSVGと診断値が一致した。carとalphaの1・4スレッド間でもSVGは一致し、診断値の違いはスレッド数だけだった（時間・出力先を除く）。

測定値、段階別時間、実行ファイルとSVGのハッシュは [測定データ](parallel-region-evaluation.json) に保存した。完全な診断値と生のSVG・ログは `/tmp/picvec-parallel/{main,single,extended}/` にある。

## 検証

- 既存の `cargo test --lib --locked --features diagnostics`: 424件成功、失敗0、既存スキップ7件。RGBA・拡大描画と報告済み微小オブジェクトの最終SVGを検証するテストを含む。
- 追加の `local_ownership_and_paint_evidence_are_identical_across_worker_counts`: 1件成功。
- `cargo check --locked --no-default-features`: 成功。既存の `paint_order::Proposal` の未使用フィールド警告が1件残る。
- 変更したRustファイルの `rustfmt --check` と `git diff --check`: 成功。全体の `cargo fmt --check` では未変更の `src/colour_fields.rs` に既存の書式差分がある。

## 再現

変更前の実行ファイルを `/tmp/picvec-parallel/baseline`、変更後を `/tmp/picvec-parallel/candidate` に保存している。出力先には新しいディレクトリを指定する。

```sh
python3 scripts/benchmark_exact.py \
  --baseline /tmp/picvec-parallel/baseline \
  --candidate /tmp/picvec-parallel/candidate \
  --output-dir /tmp/picvec-parallel-rerun \
  --threads 4 --repeats 3 --cases car wiki boy alpha key \
  --timeout-seconds 180
```

追加経路は `--threads 4 --repeats 1 --cases photo-small adaptive-small car-3passes`、単一スレッドは `--threads 1 --repeats 1 --cases car alpha` に切り替える。
