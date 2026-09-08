# クリップアート再生成・SVG化手順

この文書は、画像生成からSVG化までを再実施するときの作業メモです。
SVGDeckの通常の利用では、完成済みの
`skills/svgdeck/assets/clipart/svg/` を `svg:image` の画像リソースとして使用します。
この手順で作るSVGは、パスをスライド上で編集するものではありません。

## 生成対象のアイコン一覧

生成対象の正本は `skills/svgdeck/assets/clipart/manifest.json` です。以下は
manifestのカテゴリ順に並べた全272登録（重複を除くと268アイコン）です。
`training`、`source-code`、`search`、`home` は複数カテゴリに登録されていますが、
生成するSVGはそれぞれ1個です。manifestを更新した場合は、この一覧も同時に更新します。

- **people (16)**: `user`, `group`, `team`, `organization`, `company`, `department`, `customer`, `partner`, `guest`, `administrator`, `operator`, `developer`, `designer`, `manager`, `teacher`, `student`
- **office-business (16)**: `office`, `building`, `factory`, `store`, `warehouse`, `meeting`, `presentation`, `training`, `calendar`, `clock`, `target`, `idea`, `flag`, `trophy`, `award`, `money`
- **documents (16)**: `document`, `text`, `book`, `manual`, `notebook`, `folder`, `binder`, `clipboard`, `contract`, `invoice`, `receipt`, `report`, `template`, `form`, `label`, `tag`
- **file-formats (20)**: `pdf`, `spreadsheet`, `presentation-file`, `markdown`, `text-file`, `csv`, `tsv`, `json`, `xml`, `yaml`, `html`, `image-file`, `video-file`, `audio-file`, `archive-file`, `binary-file`, `configuration-file`, `log-file`, `source-code`, `database-dump`
- **devices (16)**: `desktop`, `laptop`, `tablet`, `smartphone`, `watch`, `display`, `printer`, `scanner`, `camera`, `microphone`, `speaker`, `terminal`, `sensor`, `iot-device`, `robot`, `robot-arm`
- **computing (16)**: `server`, `server-rack`, `mainframe`, `workstation`, `virtual-machine`, `container`, `container-cluster`, `application`, `web-application`, `mobile-application`, `desktop-application`, `service`, `api`, `batch`, `scheduler`, `workflow`
- **network (16)**: `internet`, `cloud`, `private-cloud`, `network`, `router`, `switch`, `gateway`, `firewall`, `proxy`, `load-balancer`, `vpn`, `wifi`, `dns`, `satellite`, `antenna`, `edge`
- **storage-data (16)**: `database`, `table`, `record`, `dataset`, `storage`, `object-storage`, `file-storage`, `block-storage`, `shared-storage`, `folder-storage`, `backup`, `archive`, `snapshot`, `cache`, `data-lake`, `data-warehouse`
- **communication (16)**: `mail`, `mailbox`, `chat`, `message`, `notification`, `phone`, `video-call`, `conference`, `upload`, `download`, `sync`, `broadcast`, `publish`, `subscribe`, `request`, `response`
- **development (16)**: `repository`, `branch`, `merge`, `pipeline`, `build`, `test`, `release`, `debug`, `bug`, `software-package`, `library`, `plugin`, `script`, `configuration`, `terminal-window`, `tool`
- **programming (16)**: `source-code`, `class`, `module`, `package-source`, `function`, `variable`, `constant`, `sql`, `shell`, `java`, `javascript`, `typescript`, `python`, `c`, `cobol`, `generic-language`
- **security (16)**: `security`, `lock`, `unlock`, `key`, `password`, `certificate`, `identity`, `permission`, `token`, `shield`, `encryption`, `signature`, `audit`, `privacy`, `risk`, `warning`
- **ai (16)**: `ai`, `llm`, `agent`, `brain`, `knowledge`, `memory`, `prompt`, `conversation`, `reasoning`, `search`, `embedding`, `vector`, `training`, `prediction`, `analytics`, `dashboard`
- **charts-data (16)**: `table-chart`, `bar-chart`, `line-chart`, `pie-chart`, `scatter-chart`, `dashboard-chart`, `statistics`, `trend-up`, `trend-down`, `kpi`, `report-chart`, `monitor`, `log`, `metrics`, `timeline`, `map`
- **physical-objects (20)**: `box`, `package`, `cart`, `truck`, `car`, `airplane`, `ship`, `train`, `house`, `home`, `keycard`, `usb-memory`, `hard-disk`, `optical-disc`, `battery`, `lightbulb`, `magnifier`, `camera-photo`, `envelope`, `gift`
- **common-symbols (24)**: `check`, `cross`, `warning-sign`, `question`, `information`, `plus`, `minus`, `refresh`, `search`, `settings`, `home`, `bookmark`, `star`, `heart`, `location`, `pin`, `link`, `unlink`, `filter`, `zoom`, `stop`, `play`, `pause`, `forward`

各カテゴリは意味の近いアイコンをまとめるための単位であり、1枚の生成ボードの
上限ではありません。後述の個数基準に合わせ、同じカテゴリを複数ボードへ分割します。

## 1. グループ画像を生成するプロンプト

似た意味のアイコンを同じボードにまとめ、1枚の画像から6～8個を切り出します。
同じグループが8個を超える場合は、無理に6個へ分割せず、同じ画風を保てる範囲で8～12個にまとめます。
各アイコンを切り出せるようにアイコンに100x100の枠をつけます。

### 基本プロンプト

```text
Create a clean grouped clipart board for a software architecture presentation.
Arrange {COUNT} separate icons in a precise {COLUMNS}x{ROWS} grid, one icon per
cell, with generous empty space between cells. Every icon must be fully visible,
centered in its own cell, and must not touch or overlap a neighboring icon.
Use a friendly rounded illustrated-sticker style, soft 3D volume, restrained
linear or elliptical gradients, coordinated pastel colors, and clear semantic
silhouettes. Use a uniform solid chroma-key magenta background #FF00CC only.
Do not draw borders, cards, frames, panel edges, decorative horizontal lines,
speed lines, swashes, labels, letters, numbers, logos, shadows on the
background, or any object outside the requested cells.

Icons in this board:
{ICON_1}, {ICON_2}, {ICON_3}, {ICON_4}, {ICON_5}, {ICON_6}
{OPTIONAL_MORE_ICONS}
```

### 生成時の注意

- アイコン名は具体的な英語の名詞にする（例: `desktop`, `laptop`, `database`）。
- 同じボード内で視点、光源、余白、彩度を揃える。
- 透明背景を直接指定できる場合でも、切り抜き判定が安定する単色背景を優先する。
- `desktop` と `laptop` のように隣接すると混ざりやすいものは、セル間隔を広くする。
- 人物や機器の一部が隣のセルへはみ出さないよう、各セルの境界をプロンプトで明示する。

### ネガティブプロンプト

```text
No overlapping icons, no cropped subject, no neighboring fragments, no
horizontal background stripes, no decorative lines, no card borders, no
text, no labels, no watermark, no logo, no repeated object, no extra person,
no object touching a cell boundary, no white background, no checkerboard,
no photorealism, no thin technical line-art, no hard black outline.
```

## 2. 背景除去とセル分割

1. 生成ボードを元画像として保存し、過去に切り出したSVGや過去の切り抜きを入力にしない。
2. `#FF00CC` の色キーで背景をアルファへ変換する。許容差を広げすぎず、輪郭の色を背景色として消さない。
3. 色キー除去後に、1px程度の外周収縮とデスピルを行う。
4. ボードのレイアウトから各セルの矩形を計算し、セル矩形の周囲に小さなオーバーラップを付けて切り出す。
5. 連結成分を調べ、主アイコンの中心が対象セル内にある成分を保持する。セル外の成分は、対象アイコンから十分近い場合だけ保持する。
6. セル境界をまたぐ意味のある部品は例外として指定する。たとえばDesktopの筐体はDesktopに残し、Laptop側では同じ筐体を除外する。
7. 隣のアイコンの一部、背景の線、孤立した色 speckle、アンチエイリアスの縁は削除する。
8. 切り出した画像を正方形の透明キャンバスへ正規化し、最終SVGのviewBoxを `0 0 96 96` にする。

## 3. ベクター化

1. まず透明背景付きPNGから形状を picvec を使い svg 化する
2. すべてのアイコンに、合成シルエットの外周だけへ中間灰色 `#66717C` の細い枠を付ける。内部の線や面を個別に囲わない。
