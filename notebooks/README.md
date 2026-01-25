# V-JEPA2 推論ガイド

V-JEPA2には2つの主要なモデルがあります。このドキュメントでは、それぞれのモデルの概要、使い方、実行結果について説明します。

## 目次

1. [V-JEPA 2: 動画理解モデル](#1-v-jepa-2-動画理解モデル)
2. [V-JEPA 2-AC: 世界モデル](#2-v-jepa-2-ac-世界モデル)
3. [セットアップ](#3-セットアップ)
4. [参考リンク](#4-参考リンク)

---

## 1. V-JEPA 2: 動画理解モデル

### 概要

V-JEPA 2（Video Joint Embedding Predictive Architecture 2）は、自己教師あり学習によって大量の動画データから学習したビデオエンコーダです。

**主な特徴**:
- マスクされた部分を予測する学習方式（Masked Latent Feature Prediction）
- インターネット規模の動画データで事前学習
- 動作理解・アクション予測でSOTA性能を達成

### できること

| タスク | 説明 | 評価データセット |
|--------|------|------------------|
| アクション認識 | 動画内の動作を分類 | Something-Something v2, Diving48 |
| アクション予測 | 次に何をするかを予測 | EPIC-KITCHENS-100 |
| 動画質問応答 | 動画に関する質問に回答 | MVP, TempCompass |

### 使い方

#### スクリプトの実行

```bash
# vjepa2環境をアクティベート
source /home/ubuntu/miniconda3/bin/activate vjepa2

# 推論スクリプトを実行
python -m notebooks.simple_inference
```

#### Pythonコード例

```python
import torch
from transformers import AutoModel, AutoVideoProcessor

# モデルのロード
model_name = "facebook/vjepa2-vitg-fpc64-384"
model = AutoModel.from_pretrained(model_name)
processor = AutoVideoProcessor.from_pretrained(model_name)

# 動画の読み込みと前処理
from decord import VideoReader
import numpy as np

vr = VideoReader("sample_video.mp4")
frame_idx = np.arange(0, 128, 2)  # 64フレームをサンプリング
video = vr.get_batch(frame_idx).asnumpy()
video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)

# 前処理
inputs = processor(video_tensor, return_tensors="pt")
pixel_values = inputs["pixel_values_videos"].cuda()

# 特徴抽出
model = model.cuda().eval()
with torch.inference_mode():
    features = model.get_vision_features(pixel_values)
    # features.shape: (1, 18432, 1408)
    # 18432 = (64/2) * (384/16) * (384/16) = 32 * 24 * 24 patches
```

### 実行結果

サンプル動画（ボウリングボールをチューブに入れる動作）に対する分類結果:

```
============================================================
RESULTS: Top 5 Predicted Actions
============================================================

Sample video: sample_video.mp4
(Bowling video from Kinetics dataset)

  1. Putting [something] into [something]
     Confidence: 37.69%

  2. Stuffing [something] into [something]
     Confidence: 16.05%

  3. Putting [something] onto [something]
     Confidence: 15.20%

  4. Putting [number of] [something] onto [something]
     Confidence: 6.50%

  5. Failing to put [something] into [something]
     Confidence: 5.08%
```

**解釈**: モデルは「何かを何かに入れる」動作を正しく認識しています。

### 出力の構造

```
入力: 動画 (64フレーム × 384×384 RGB)
  ↓
エンコーダ (ViT-g/16)
  ↓
出力: パッチ特徴 (18,432 patches × 1,408 dimensions)
  ↓
分類器 (Attentive Classifier)
  ↓
出力: クラス確率 (174クラス for SSv2)
```

### 利用可能なモデル

| モデル | パラメータ数 | 解像度 | HuggingFace |
|--------|-------------|--------|-------------|
| ViT-L/16 | 300M | 256 | facebook/vjepa2-vitl-fpc64-256 |
| ViT-H/16 | 600M | 256 | facebook/vjepa2-vith-fpc64-256 |
| ViT-g/16 | 1B | 256 | facebook/vjepa2-vitg-fpc64-256 |
| ViT-g/16 | 1B | 384 | facebook/vjepa2-vitg-fpc64-384 |

---

## 2. V-JEPA 2-AC: 世界モデル

### 概要

V-JEPA 2-AC（Action-Conditioned）は、V-JEPA 2を基に少量のロボットデータで追加学習した**世界モデル**です。

**世界モデルとは**: 「あるアクションを取ったら、世界（環境）がどう変化するか」を予測するモデル。

**主な特徴**:
- 現在の状態とアクションから未来の状態を予測
- 少量のロボットデータ（約50時間）で学習
- 新しい環境でもタスク固有の訓練なしで動作

### できること

| タスク | 説明 | 成功率 |
|--------|------|--------|
| Reach | 目標位置に到達 | 100% |
| Grasp (Cup) | カップを掴む | 60% |
| Grasp (Box) | 箱を掴む | 20% |
| Pick-and-Place (Cup) | カップを移動 | 80% |
| Pick-and-Place (Box) | 箱を移動 | 50% |

### 使い方

#### スクリプトの実行

```bash
# vjepa2環境をアクティベート
source /home/ubuntu/miniconda3/bin/activate vjepa2

# 世界モデルデモを実行
python /home/ubuntu/vjepa2/notebooks/world_model_demo.py
```

#### Pythonコード例

```python
import torch

# モデルのロード（PyTorch Hub）
encoder, predictor = torch.hub.load(
    "facebookresearch/vjepa2",
    "vjepa2_ac_vit_giant"
)
encoder = encoder.cuda().eval()
predictor = predictor.cuda().eval()

# 入力データ
# - current_frame: 現在のフレーム特徴 (1, 256, 1408)
# - action: ロボットアクション (1, 1, 7) [x, y, z, rx, ry, rz, gripper]
# - state: ロボット状態 (1, 1, 7)

# 未来の予測
with torch.inference_mode():
    predicted_features = predictor(current_frame, action, state)
    # predicted_features.shape: (1, 256, 1408)
```

### 実行結果

#### 軌道データ

![trajectory_frames.png](trajectory_frames.png)

- **Frame 1 (Current)**: ロボットアームの現在位置
- **Frame 2 (Target/Future)**: アクション実行後の目標位置

#### エネルギーランドスケープ

![energy_landscape.png](energy_landscape.png)

**解釈**:
- **色が濃い部分**: 予測がターゲットに近い（良いアクション）
- **赤い星**: 実際のアクション (Ground Truth: x=0.092, z=0.084)
- **シアンの丸**: モデルが最適と判断したアクション (x=0.100, z=0.100)

モデルの予測がGround Truthに近いことがわかります。

#### 推論精度

```
予測精度:
  - Mean Absolute Error: 0.4251
  - Cosine Similarity: 0.7314 (73% 類似)
```

### 動作原理

```
┌─────────────────────────────────────────────────────────────┐
│                    V-JEPA 2-AC の動作                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [現在のフレーム] + [アクション] + [ロボット状態]             │
│         │              │              │                     │
│         └──────────────┼──────────────┘                     │
│                        ↓                                     │
│              ┌─────────────────┐                            │
│              │   Encoder       │  フレームを特徴に変換        │
│              │   (ViT-g/16)    │                            │
│              └────────┬────────┘                            │
│                       ↓                                      │
│              ┌─────────────────┐                            │
│              │   Predictor     │  未来の特徴を予測           │
│              │   (305M params) │                            │
│              └────────┬────────┘                            │
│                       ↓                                      │
│              [予測された未来の特徴]                          │
│                       ↓                                      │
│         ┌─────────────────────────────┐                     │
│         │  エネルギー計算              │                     │
│         │  E = |予測 - 目標|          │                     │
│         └─────────────┬───────────────┘                     │
│                       ↓                                      │
│         [エネルギーが最小のアクションを選択]                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 計画（Planning）の仕組み

世界モデルを使ったロボット制御:

1. **目標画像**を与える（例: カップを棚に置いた状態）
2. 様々な**アクション候補**を生成
3. 各アクションで**未来を予測**
4. 目標に最も近い予測を与えるアクションを**選択**
5. アクションを**実行**
6. 1-5を繰り返す（Model Predictive Control）

---

## 3. セットアップ

### 環境構築

```bash
# Conda環境の作成
conda create -n vjepa2 python=3.12
conda activate vjepa2

# パッケージのインストール
cd /home/ubuntu/vjepa2
pip install -e .

# 追加パッケージ（可視化用）
pip install matplotlib
```

### 必要なファイル

| ファイル | 用途 | 自動ダウンロード |
|----------|------|------------------|
| sample_video.mp4 | サンプル動画 | Yes |
| ssv2_classes.json | クラスラベル | Yes |
| ssv2-vitg-384-64x2x3.pt | 分類器重み | Yes |
| franka_example_traj.npz | ロボット軌道 | No (リポジトリに含まれる) |

### ファイル構成

```
notebooks/
├── README.md                 # このファイル
├── simple_inference.py       # 動画理解デモ
├── world_model_demo.py       # 世界モデルデモ
├── vjepa2_demo.ipynb         # オリジナルノートブック
├── energy_landscape_example.ipynb  # 世界モデルノートブック
├── franka_example_traj.npz   # ロボット軌道データ
├── trajectory_frames.png     # 軌道可視化結果
└── energy_landscape.png      # エネルギーランドスケープ結果
```

---

## 4. 参考リンク

- [論文 (arXiv)](https://arxiv.org/abs/2506.09985)
- [公式ブログ](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks)
- [HuggingFace Collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6)
- [GitHub リポジトリ](https://github.com/facebookresearch/vjepa2)

---

## 比較表

| 項目 | V-JEPA 2 | V-JEPA 2-AC |
|------|----------|-------------|
| **目的** | 動画を理解する | 未来を予測する |
| **入力** | 動画フレーム | フレーム + アクション + 状態 |
| **出力** | 特徴ベクトル / クラス | 未来の特徴ベクトル |
| **主な用途** | 動画分類、アクション認識 | ロボット制御、計画立案 |
| **学習データ** | インターネット動画 | + 少量のロボットデータ |
| **パラメータ数** | ~1B (エンコーダ) | ~1.3B (エンコーダ + プレディクタ) |
