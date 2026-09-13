# 可視性検証の追加高速化（2026-09-13）

前回の精度維持の高速化を適用済みの版を基準に、可視性検証の3項目を実装した。変更箇所は `src/visibility.rs`。画質設定、候補の採用順、元サイズと4倍のRGBA完全一致という削除条件は維持した。

- 候補IDごとのSVGツリー全走査を、1回の走査に集約した。隔離グループ内も従来のID検索と同じ範囲を調べる。
- タイル内の描画済み前半を保持し、候補ごとに残りを元と同じ順序で描画する。前半はフレームバッファをそのままコピーし、透過の丸めを変える別画像の合成は行わない。候補の処理順が単調なので、採用済み削除を反映しながら前半を進められる。
- キャッシュ満杯時の全消去を、最も長く使われていない1件の入れ替えに変更した。履歴は固定容量で保持する。

CPU上限は変更していない。既定は論理CPU数の半分、最大4スレッド。明示した `--threads` は従来どおり利用可能CPU数の範囲で尊重する。今回の変更には並列処理の追加がない。

キャッシュは最大64件、それぞれ基準画像と描画済み前半を持つため最大128枚。画素バッファの上限は従来どおり32 MiBと作業用1枚（最大256 KiB）。追い出した画像を解放してから次を確保する。これはタイルキャッシュの上限であり、SVGツリーなどを含むプロセス全体のメモリ上限ではない。

## 可視性検証だけの測定

保存済みの出力SVGを入力に、テスト専用に保存した旧実装 `src/visibility_reference.rs` と比較した。これらは削除処理済みのSVGで、両版とも追加削除は0件。変換全体で最初に行う削除処理は次節で別に比較している。

各3回は実行順を交互にした中央値。大判2件は各1回。変更前後は同時実行していない。SVG解析を含む `prune` 呼び出しの時間であり、画像からSVGまでの変換全体の時間ではない。

| ケース | 回数 | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|---:|
| car | 3 | 0.917976 | 0.660123 | 28.1% |
| wiki | 3 | 0.594621 | 0.236394 | 60.2% |
| key | 3 | 0.092907 | 0.051351 | 44.7% |
| alpha | 3 | 0.000059 | 0.000065 | -9.1% |
| photo-small | 3 | 4.733434 | 1.191071 | 74.8% |
| cliparts | 3 | 0.085742 | 0.031492 | 63.3% |
| photo-large | 1 | 83.785687 | 46.188330 | 44.9% |
| sheet-large | 1 | 4.397415 | 0.356786 | 91.9% |

8種類・計20回で、SVG文字列と削除件数の全フィールドが完全一致した。alphaは数十マイクロ秒と短く、この測定から速度差を評価しない。photo-largeは `sample/output/viewport2.svg`、sheet-largeは `sample/output/cliparts-6x6.svg`。大判の元ラスター画像からの変換全体は今回測定していない。

## 画像からSVGまでの変換全体

`--release --features diagnostics`、4スレッド、各1回、変更前→変更後の順。プロセス起動とファイル入出力を含む。負の短縮率は時間の増加を示す。

| ケース | 変更前（秒） | 変更後（秒） | 短縮率 |
|---|---:|---:|---:|
| car | 55.015 | 57.798 | -5.1% |
| wiki | 35.943 | 36.780 | -2.3% |
| photo-small | 30.320 | 26.470 | 12.7% |
| key | 26.782 | 27.230 | -1.7% |
| alpha | 0.890 | 0.989 | -11.1% |
| adaptive-small | 58.486 | 59.495 | -1.7% |

全6ケースでSVGのSHA-256と診断値が一致した。除外した診断値は経過時間と出力パスのみ。adaptive-smallは初期最大寸法512で適応的高精細化を有効にしたケース。

車とロゴでは総時間が増加した一方、今回変更した処理を含む `final-svg` はそれぞれ2.366→2.219秒、0.772→0.356秒だった。車では未変更の `paint-fitting` が1.898秒、`interior-shading` が0.791秒増加し、ロゴでは未変更の `source-supported-paint-merge` が1.238秒増加した。単発測定なので、総時間の差を安定した改善率や悪化率とは断定しない。写真の `final-svg` は6.614→2.065秒だった。

## 検証と再現

- `RAYON_NUM_THREADS=4 cargo test --release --lib --features diagnostics -- --test-threads=1`: 420 passed、0 failed、7 ignored。無視されたもののうち今回追加した測定テストは、その後8入力で明示的に実行して成功した。
- 旧実装との比較テストで、RGBAバッファ、削除判定、SVGを確認。透過、グラデーション、細線、幾何クリップ、タイル境界、連続削除、キャッシュ容量1・3・64を含む。描画済み前半の比較は1・4・8倍で実施した。
- `rustfmt --check --edition 2021 src/visibility.rs src/visibility_reference.rs`、`git diff --check` が成功。
- `cargo check --no-default-features` が成功。既存の `paint_order::Proposal` の未使用フィールド警告1件あり。

可視性検証の測定例（既定3回、必要なら `PICVEC_VISIBILITY_BENCH_REPEATS=1`）:

```sh
PICVEC_VISIBILITY_BENCH_INPUTS="$PWD/sample/output/viewport1.svg" \
PICVEC_VISIBILITY_BENCH_OUTPUT=/tmp/picvec-visibility-stage.json \
RAYON_NUM_THREADS=4 CARGO_BUILD_JOBS=4 \
cargo test --release --lib --features diagnostics \
  visibility::tests::benchmark_emitted_svgs_against_full_redraw_reference \
  -- --exact --ignored --nocapture --test-threads=1
```

全体測定:

```sh
python3 scripts/benchmark_exact.py \
  --baseline /tmp/picvec-visibility-perf/baseline \
  --candidate /tmp/picvec-visibility-perf/candidate \
  --output-dir /tmp/picvec-visibility-perf/pipeline-rerun \
  --threads 4 --repeats 1 \
  --cases car wiki photo-small key alpha adaptive-small \
  --timeout-seconds 180
```

基準バイナリのSHA-256: `9826caa01b9c52a7a292ca5f09c2201ad6331c4ba562c56486563d2eca7b1644`。
変更後: `21b0a9c644093e41e7fbec6f8a6dfff96cc8b8a2aef757cbec2929acd30aeae5`。

詳細データは [visibility-performance-2026-09-13.json](visibility-performance-2026-09-13.json)。一時ログと比較SVGは `/tmp/picvec-visibility-perf/` に保存した。前回の測定は [performance-2026-09-13.md](performance-2026-09-13.md) を参照。
