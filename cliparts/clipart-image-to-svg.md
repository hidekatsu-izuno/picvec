# クリップアート再生成・SVG化手順

この文書は、画像生成からSVG化までを再実施するときの作業メモです。

生成対象のアイコンと既定の並び順は、同じディレクトリの [manifest.json](manifest.json) の
`categories` 配列と各カテゴリの `items` 配列で管理します。
生成時は類似性に応じて並べ替えてよいものとし、実際の配置は後述の対応表で管理します。
`{key}` は `items` の値です。似た対象でも異なるキーを統合・省略せず、各キーを全画像で1回ずつ生成します。

## 生成手順

1. 似たアイコンとなりそうな対象を分類し、各分類の代表などから表現の異なる36アイコンを選びます。生成前にセルとキーの対応表を保存し、6列×6行の正方形グリッドに1セル1個ずつ配置した基準画像を imagegen で生成して `cliparts/guide/base.png` として保存します。
2. 生成した基準画像をユーザーに提示しこのまま進めてよいか了承を得ます。
3. 基準画像に選ばれなかった対象を最大36アイコンずつに分けます（似たアイコンは可能な範囲で同じグループにします）。各画像の生成前に対応表を保存し、6列×6行の正方形グリッドで imagegen により生成して `cliparts/guide/group-{n}.png` として保存します。`{n}` は1からの連番です。基準画像を画風・視点・光源・スケールのガイドとして添付し、アイコンの内容と配置は各グループの対応表に従います。最終画像が36個未満なら、左上から順に配置し、残りのセルは完全に透明にします。
4. すべての画像が出力されたら、対応表どおりの配置と未使用セルの透明性を確認します。不一致があれば修正してから、基準画像を含む各画像を6列×6行のセル境界で切り出し、使用セルだけを `cliparts/input/{key}.png` として保存します。
5. picvec を使い SVG に変換し `cliparts/output/{key}.svg` として保存します。

### セルとキーの対応表

各画像と同名のJSONファイル（`cliparts/guide/base.json`、`cliparts/guide/group-{n}.json`）に、
`rows` として6行×6列の配列を保存します。各要素はキー文字列、未使用セルは `null` とします。
この対応表を生成プロンプトと切り出し時の命名の両方に使用します。
現行manifestの256アイコンでは、基準画像36個、追加画像6枚×36個、最終画像4個となります。

### プロンプト

`{COUNT}` は対応表の使用セル数（基準画像は36）、`{ICON_ROWS_FROM_MAPPING}` は
対応表を行ごとに展開した6行×6項目（`null` は `EMPTY`）に置き換えます。
`{STYLE_REFERENCE_INSTRUCTION}` は基準画像の生成時には省略し、追加画像では次の文に置き換えます。

> Use the attached base image as a reference for style, viewpoint, lighting, and scale. Follow the assignments below for icon content and placement; do not copy the reference image's icon selection.

```text
Create ONE square clipart contact sheet for a software architecture presentation.

Use the largest square output size available in ImageGen and set:

`transparent: true`

Create exactly {COUNT} distinct icons in a precise 6-column × 6-row grid with 36 equal square cells.

{STYLE_REFERENCE_INSTRUCTION}

Map the icon assignments strictly left-to-right, then top-to-bottom:

* cells 1–6 → Row 1, Columns 1–6
* cells 7–12 → Row 2, Columns 1–6
* continue identically through
* cells 31–36 → Row 6, Columns 1–6

Place exactly one requested icon in each assigned cell. Leave cells marked EMPTY completely transparent, without any artwork or text. A requested group or collection counts as one composition and must remain entirely within its assigned cell.

Center every icon and keep it fully inside its cell with at least 10% transparent padding on every side. Keep all icons visually consistent in scale.

Use a friendly rounded illustrated-sticker style with:

* soft 3D volume
* restrained linear or elliptical gradients
* coordinated pastel colors
* crisp, clearly separated shapes
* strong recognizable silhouettes
* consistent viewpoint, lighting, scale, and detail

Prefer simple, meaningful shapes that remain recognizable at small size. Do not rely on text or logos.

All space outside the icon artwork must be genuine alpha transparency.

### Exact assignments

Each of the six rows below contains exactly six entries, either an icon key or EMPTY. Row names and EMPTY are instructions only and must not appear in the image.

{ICON_ROWS_FROM_MAPPING}

### Negative prompt

No missing icons, no duplicate icons, no extra icons, no reordered cells,
no merged cells, no uneven grid, no overlapping icons, no cropped subjects,
no neighboring fragments, no object touching a cell boundary, no visible grid
lines, no horizontal background stripes, no decorative lines, no card borders,
no text, no labels, no row or column numbers, no watermark, no logo,
no white background, no checkerboard, no background shadows, no photorealism,
no thin technical line-art, no hard black outline.
```
