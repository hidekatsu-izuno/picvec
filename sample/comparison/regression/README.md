# 全サンプルの再生成・回帰確認

**結論：全10件を再生成しましたが、品質の悪化なしとは判定できません。**

6件のSVGは変更前とバイト単位で一致しました。変更された4件は全体のSSIMが改善していますが、静物画の網目、図面の線・文字などに局所的な悪化があります。viewport1は平均色誤差も微増しました。

今回の作業では変換ロジックを変更せず、現行の処理で全SVGを再生成して検証しました。

## 処理時間・容量

基準は `c659d51d95c2a6c24c4eeaac77c1ea95d48883d1`。両方を同じRust 1.95.0、release、diagnosticsでビルドし、4スレッドで逐次実行しました。ビルド・画像描画と計測を重ねていません。時間はプロセス起動から終了までです。背景除去は従来どおりcliparts-6x6のみ有効です。

car・静物画・viewport1は実行順を反転しながら各3回計測した中央値、他は初回1回の値です。1回の数値は速度改善・悪化の確定値ではありません。数%の差には環境のばらつきが含まれます。

| サンプル | 回数 | 時間：旧 → 新（秒） | 時間差 | SVG容量差 | SVG完全一致 |
| --- | ---: | ---: | ---: | ---: | :---: |
| [booster-layout](../../output/booster-layout.svg) | 1 | 235.43 → 215.66 | -8.4% | +20.9% | いいえ |
| [boy_and_turtle](../../output/boy_and_turtle.svg) | 1 | 5.41 → 5.46 | +0.9% | +0.0% | はい |
| [car](../../output/car.svg) | 3 | 22.49 → 22.70 | +0.9% | +0.0% | はい |
| [catwhale](../../output/catwhale.svg) | 1 | 33.56 → 27.22 | -18.9% | -6.6% | いいえ |
| [cliparts-6x6](../../output/cliparts-6x6.svg) | 1 | 355.84 → 351.94 | -1.1% | +0.0% | はい |
| [cliparts](../../output/cliparts.svg) | 1 | 39.57 → 39.95 | +0.9% | +0.0% | はい |
| [vectorization-stress-still-life](../../output/vectorization-stress-still-life.svg) | 3 | 75.03 → 76.97 | +2.6% | -1.3% | いいえ |
| [viewport1](../../output/viewport1.svg) | 3 | 8.98 → 8.46 | -5.9% | -0.4% | いいえ |
| [viewport2](../../output/viewport2.svg) | 1 | 73.61 → 74.06 | +0.6% | +0.0% | はい |
| [wikipedia_logo_1_0](../../output/wikipedia_logo_1_0.svg) | 1 | 17.32 → 17.18 | -0.8% | +0.0% | はい |

## 画質

完全なSVGを入力画像と同じ寸法で描画し、白背景・黒背景に合成して原画像と比較しました。下表は白背景でのRGB平均絶対誤差（0〜255、小さいほど良い）と輝度SSIM（大きいほど良い）です。黒背景とalpha誤差もJSONに保存しています。SSIMは11×11の均等重み窓、Rec.709輝度係数、K1=0.01、K2=0.03で計算しています。

| サンプル | RGB MAE：旧 → 新 | SSIM：旧 → 新 |
| --- | ---: | ---: |
| booster-layout | 4.5003 → 4.2039 | 0.94411 → 0.94782 |
| catwhale | 3.9010 → 3.6557 | 0.93076 → 0.93680 |
| vectorization-stress-still-life | 12.1163 → 12.0747 | 0.78270 → 0.78317 |
| viewport1 | 12.6516 → 12.6597 | 0.72278 → 0.72456 |

平均誤差の増加が大きい64×64ピクセルのタイルを各画像から3箇所選び、SVG自体を4倍で描画しました。比較画像の左は原画像の最近傍拡大、中央は旧SVG、右は新SVGです。成功例だけを選んだ比較ではありません。座標は原画像基準です。

- **静物画**：座標(256,576)付近で網目が薄れ、暗い塊にまとまっています。平均誤差が改善していても、この局所変化は悪化です。
- **booster-layout**：座標(1152,1856)付近の線、(1984,896)付近の文字、(2624,384)付近の記号が太くなっています。全体および文字領域の平均改善とは別に、こうした退行が残ります。
- **catwhale**：文字の改善に加え、文字以外の陰影・輪郭にも変化があります。全領域が改善したわけではありません。
- **viewport1**：平均色誤差が微増し、雪と岩の境界などで形状が変わっています。SSIMの改善だけで劣化なしとは判断していません。

- booster-layout：[全体比較](booster-layout-overview.png) / [悪化側の局所比較・SVG実描画4倍](booster-layout-worst-details.png)
- catwhale：[全体比較](catwhale-overview.png) / [悪化側の局所比較・SVG実描画4倍](catwhale-worst-details.png)
- vectorization-stress-still-life：[全体比較](vectorization-stress-still-life-overview.png) / [悪化側の局所比較・SVG実描画4倍](vectorization-stress-still-life-worst-details.png)
- viewport1：[全体比較](viewport1-overview.png) / [悪化側の局所比較・SVG実描画4倍](viewport1-worst-details.png)

## 確認範囲

- 全10件で旧バイナリの出力がコミット済みSVGと完全一致し、比較基準を確認しました。
- 6件の完全一致にはRGBA画像と背景除去画像が含まれ、透明部分も同一です。変更された4件はRGBとalphaの両方を比較しました。
- 全10件でmask要素・属性、埋め込み画像、text要素、group opacity、fill-opacity=0がないことを確認しました。
- 再計測した出力も各バージョン内でバイト単位で一致しました。
- catwhaleとbooster-layoutの再生成結果は、前回の[OCR評価](../ocr/README.md)の対象SVGと完全一致しています。
- この検証では速度・品質を測定しました。局所的な品質退行を修正するロジック変更は行っていません。

[生の計測値・ハッシュ・画質指標・確認結果](results.json)
