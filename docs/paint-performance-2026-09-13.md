# 塗り推定・領域統合の追加高速化（2026-09-13）

比較基準は `9e67747`。前回の可視性検証の高速化までを含む。この変更ではCPU上限、サンプル数、画質設定、誤差しきい値、候補の採用順を維持した。

## 変更

- `src/edge.rs`: 256画素以下のOKLab変換を逐次処理する。小さな塗り候補ごとに発生するRayonの分割・同期を省く。257画素以上の既存の並列処理と、各画素の変換式・並び順は維持した。
- `src/gradient.rs`: 中央値のための全並べ替えを順位選択に変更した。偶数個では下側の最大値も求め、従来と同じ加算・乗算を行う。符号付きゼロを含む `total_cmp` の順序を維持した。中央値を使う箇所は値の取得が目的で、並べ替え後の配列への依存がないことを確認した。
- 同ファイルの領域統合判定: 領域単独の検証で求めた色差配列を、統合後の検証にも再利用する。以前は塗りの描画・OKLab変換・色差を再計算していた。パーセンタイル計算は配列を並べ替えるため、その作業にはコピーを使い、全体の平均を求める加算順を変えない。

新しい並列処理や永続キャッシュは追加していない。実行スレッド数の設定コードは未変更。既定は論理CPU数の半分、最大4で、明示した `--threads` も従来どおり尊重する。

## 調査

`tests/color_batch_performance.rs` で色変換を256回繰り返し、1・4スレッド、単独処理・外側で領域を並列化した場合を測った。これは明示的に実行する無視指定の測定テストで、通常のテストに時間の合否条件は加えていない。

予備測定では4スレッドで32画素の単独バッチ256回が並列0.003876秒・逐次0.000278秒、256画素では0.007171秒・0.002326秒だった。768画素では並列が速い測定があったため、小規模バッチに限定した。これは1回のマイクロベンチマークであり、変換全体の改善率を示すものではない。変更前後の全ログはJSONに保存した。

## 変換全体の実測

Rust releaseビルド、`--features diagnostics`。変更前後を同時実行せず、通常ケースは実行順を入れ替えて各3回。表は中央値で、プロセス起動・ファイル入出力も含む。負の短縮率は時間の増加を示す。追加ケースは各1回なので、安定した改善率とは断定しない。

### main（4スレッド）

| ケース | 回数 | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|---:|
| car | 3 | 54.947 | 41.378 | 24.7% |
| wiki | 3 | 34.656 | 28.887 | 16.6% |
| photo-small | 3 | 26.525 | 20.662 | 22.1% |

### additional（4スレッド）

| ケース | 回数 | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|---:|
| key | 1 | 25.039 | 19.788 | 21.0% |
| alpha | 1 | 0.891 | 0.887 | 0.4% |
| adaptive-small | 1 | 58.287 | 49.405 | 15.2% |

### single-thread（1スレッド）

| ケース | 回数 | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|---:|
| photo-small | 1 | 26.876 | 26.578 | 1.1% |
| alpha | 1 | 1.361 | 1.363 | -0.1% |

carは車、wikiはWikipediaロゴ、photo-smallは640×426の写真。keyはボタン画像の背景除去、alphaは透過画像、adaptive-smallはclipartsの初期最大寸法512で適応的高精細化を有効にしたケース。

### 主要工程（各3回の中央値、4スレッド）

工程時間は診断ログの値。並列処理の待ち時間などを含むため、各変更単独の寄与率を示すものではない。

| ケース | 工程 | 変更前（秒） | 変更後（秒） |
|---|---|---:|---:|
| car | paint-aware-merge | 1.973 | 1.679 |
| car | paint-fitting | 2.388 | 2.291 |
| car | source-supported-paint-merge | 18.011 | 6.250 |
| car | interior-shading | 5.871 | 5.192 |
| wiki | paint-aware-merge | 0.823 | 0.775 |
| wiki | paint-fitting | 2.353 | 2.436 |
| wiki | source-supported-paint-merge | 12.222 | 6.335 |
| wiki | interior-shading | 2.539 | 2.220 |
| photo-small | paint-aware-merge | 3.001 | 0.482 |
| photo-small | paint-fitting | 3.768 | 1.899 |
| photo-small | source-supported-paint-merge | 1.318 | 0.536 |
| photo-small | interior-shading | 0.624 | 0.424 |

## 精度と検証

計14組の全比較でSVGのSHA-256と診断値が一致した。診断値から除外したものは経過時間と出力パスのみ。同一SVGなので透過と拡大時の描画も変更されない。写真・透過画像では1スレッドと4スレッドのSVGも一致した。1スレッドでは時間差が小さく、今回の大きな改善は4スレッドでの測定結果である。

- `RAYON_NUM_THREADS=4 CARGO_BUILD_JOBS=4 cargo test --release --lib --features diagnostics -- --test-threads=1`: 423 passed、0 failed、7 ignored（既存の大規模テスト・可視性測定）。
- 新しいテストでは、色変換の切り替え境界を含むビット単位の一致、中央値と全並べ替えの一致、再利用前の統合判定との採用・2種の棄却結果の一致を確認した。
- 色変換の無視指定ベンチマークを変更前後で明示的に実行して成功した。
- `cargo check --no-default-features` が成功。既存の `paint_order::Proposal` の未使用フィールド警告1件あり。
- 変更したRustファイルの `rustfmt --check` と `git diff --check` が成功。

再現例:

```sh
CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 cargo test --release --features diagnostics \
  --test color_batch_performance -- --ignored --nocapture --test-threads=1

python3 scripts/benchmark_exact.py \
  --baseline /tmp/picvec-paint-perf/baseline \
  --candidate /tmp/picvec-paint-perf/candidate \
  --output-dir /tmp/picvec-paint-perf/rerun \
  --threads 4 --repeats 3 --cases car wiki photo-small --timeout-seconds 180
```

baselineのSHA-256: `21b0a9c644093e41e7fbec6f8a6dfff96cc8b8a2aef757cbec2929acd30aeae5`。

candidateのSHA-256: `a6ae42f17b8df23c68182a62cac9efbb67c1c8d4ca848bd1cd5e19ceb7d4557d`。

詳細データは [paint-performance-2026-09-13.json](paint-performance-2026-09-13.json)。比較SVGとログは `/tmp/picvec-paint-perf/`。
