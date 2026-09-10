処理最適化の実装・検証記録（2026-09-10）

レビュー時点の未コミット変更を含むソースを基準として、候補の種類、探索順、誤差の計算順、採用閾値を維持する最適化を段階的に実装した。既存のサンプルSVG・比較画像は更新していない。

実装した変更は以下。

| 段階 | 内容 | 同一結果を維持する方法 |
| --- | --- | --- |
| 1 | 中間の品質評価を削除 | 最終SVGで必ず上書きされていた評価だけを除去 |
| 1 | 採用不能な統合候補の後処理・ヒープ登録を省略 | 有限の候補同士の順序を維持 |
| 1 | 同じPaint・方向の候補を重複排除 | 最初の出現を残し、候補の順序を維持 |
| 1 | ridgeの二重OKLab変換を削除 | 検出と後処理に同じ変換結果を渡す |
| 1 | percentileを順位選択に変更 | `total_cmp` と隣接2順位の補間式を維持 |
| 2 | Paint fittingの色・境界情報を後続のmergeへ引き継ぐ | 未加工の境界情報を保存し、ラベル更新後のroundでは再構築 |
| 2 | quiet domainの連結成分を閾値ごとに共有 | fitting用の画素順と領域拡張用のBFS順を別々に保持 |
| 3 | 残差部分集合探索の観測値を共有 | 全組合せとレイヤーの合成順を維持し、色・不透明度だけを再利用 |
| 3 | 残差の傾き検証で元画像とベースのOKLabを共有 | 必要な画素を遅延評価し、候補間で元の観測値を再利用 |
| 3 | 試行レンダリング間のPaintパス最適化を共有 | 入力パス文字列をキーにし、通常境界とocclusion境界を混同しない |
| 4 | 輪郭候補間で原輪郭の空間索引を共有 | セル分割、近傍順、距離式、距離の集計順を維持 |
| 4 | 統合候補内で評価用OKLabを共有 | 元と同じサンプル列・部分列で評価 |
| 4 | bilateral filterの一時weight配列を再利用 | 81近傍の順序、SIMD exp、画素ごとの加算順を維持 |
| 4 | 診断の時間区間を分離 | 完全一致統合・レイヤー簡略化・残差補正・hierarchyを個別表示 |
| 5 | 残差追加フィットで同じ下地の色を共有 | 1 round中は不変の下地をキャッシュし、追加レイヤーだけを評価 |

残差の観測値共有は `src/gradient_residual.rs` に分離した。元画像・ベースの遅延キャッシュはregionごと、SVGのパスキャッシュはconversionごとに作成・解放し、別の画像や更新後のPaintに持ち越さない。公開されている `gradient::fit_all` の戻り値は変更していない。

レビューで指摘した「簡略化後の検証」のコメントは、実際に残存するLayered Paintだけを検証することが分かる説明に修正した。検証対象を広げる変更、coherenceや色・面積の閾値変更、候補探索数の削減、近似フィルターへの置換は、結果同一の条件に合わないため実施していない。棄却される候補の探索を無条件に省略することもしていない。

比較には修正前のソースと実行ファイルを `/tmp/picvec-equivalence/baseline` に保存した。各段階の実行ファイルも別ディレクトリに保存し、生成先を `/tmp` としてSHA-256と診断JSONを比較した。比較対象から除いた診断フィールドは、最上位の `elapsed_seconds` と `output` だけ。領域数、構造線、Paintの種類、adaptiveの採否、品質値などは一致判定に含めた。

比較ケースは `car`、`boy`、`photo`（viewport2）、`wiki`、`alpha`（cube）、`key`（round buttons）、`car-quality`、`car-3passes`、`adaptive`（cliparts-6x6）の9種類。通常の4ワーカー、CLI既定の処理解像度を使い、keyとadaptiveにはクロマキー除去を指定する。

再実行用の比較スクリプトは `scripts/benchmark_exact.py`。両実行ファイルは同じrelease設定で、`--features diagnostics` を付けてビルドする。出力先には未作成のディレクトリを指定する。

```bash
python3 scripts/benchmark_exact.py \
  --baseline /tmp/picvec-equivalence/baseline/picvec \
  --candidate /tmp/picvec-equivalence/step5/picvec \
  --output-dir /tmp/picvec-equivalence/paired-final \
  --cases car wiki boy alpha --repeats 3 --threads 4
```

スクリプトはbaseline/candidateを逐次実行し、2巡目は順序を逆転する。全試行でSVGと診断値の一致を検証し、各処理時間と中央値を `results.json` に記録する。出力が変わった場合には途中で失敗する。時間の比較時は、他の変換・ビルド・テストを同時に実行しない。

段階ごとの比較は、段階1でcar・boy、段階2でcar・boy・wiki・alpha・key、段階3でcar・wiki・car-quality、段階4で全9ケースを実行し、いずれも基準版のSVGと診断値に一致した。段階間の処理時間は他の確認処理と並行実行した値なので、速度の比較には使用しない。

最終版（段階5）も全9ケースでSVGがバイト単位で一致し、上記2フィールド以外の診断値も一致した。adaptiveケースは36領域の再変換を含む。入力とソースのSHA-256、各段階の比較ケースは `docs/processing-optimization-results.json` に記録した。目視上の類似や平均画質だけによる判定ではない。

最終版のテスト結果：

- `mise exec -- cargo test --release --locked --lib --features diagnostics -- --test-threads=4`：328件成功、0件失敗、5件ignore（76.87秒）。
- `mise exec -- cargo check --locked --all-targets --no-default-features`：成功。
- 新規の差分テストでは、順位選択と全sortによるpercentile、5層の全32部分集合のMSE、遅延キャッシュと元の傾き検証を比較した。既存の空間索引と全走査の距離一致テストも成功した。

計測環境はAMD Ryzen AI 9 465、WSL2/Linux x86_64、Rust 1.95.0。同じrelease設定・4ワーカーで比較する。これらの入力に対する出力一致の確認であり、全入力・全環境に対する保証ではない。

独立した速度計測の結果（各版3回の実時間中央値）：

| 入力 | 修正前 | 修正後 | 時間短縮率 |
| --- | ---: | ---: | ---: |
| car | 57.910秒 | 47.001秒 | 18.8% |
| Wikipediaロゴ | 33.525秒 | 31.865秒 | 5.0% |
| boy_and_turtle | 4.045秒 | 3.946秒 | 2.4% |
| cube-alpha | 0.911秒 | 0.863秒 | 5.3% |

4ケース×2実行ファイル×3回の全24試行で、SVGと診断値の一致を確認した。各ペアでも修正後の方が短時間だった。ただし短時間の入力の差は小さく、測定揺らぎの影響がある。写真・クロマキー・adaptive等は出力一致を確認したが、独立した反復速度計測は行っていないため、その高速化率は示さない。メモリ使用量の改善率も計測していない。

各試行の処理時間・段階別時間、実行ファイルのSHA-256、比較条件は [processing-optimization-results.json](processing-optimization-results.json) に保存した。結果が変わる判定基準の見直しは残しているが、重複計算の削減と厳密な計算の再利用により、確認した入力では結果を維持して処理時間を短縮できた。
