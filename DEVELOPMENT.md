# Development Workflow

このリポジトリは公式 [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) のフォークです。

## ブランチ構成

```
公式 (upstream)              自分のfork (origin)
facebookresearch/vjepa2      muchan23/vjepa2-dev
        │                           │
        │  fetch (読み取り専用)      │
        │ ────────────────────>     │
        │                           ├── main (公式と同期用)
        │                           └── dev  (開発用)
```

- `main`: 公式リポジトリと同期するためのブランチ
- `dev`: 自分の開発を行うブランチ

## リモート設定

```bash
# 現在の設定を確認
git remote -v

# 期待される出力:
# origin    git@github.com:muchan23/vjepa2-dev.git (fetch)
# origin    git@github.com:muchan23/vjepa2-dev.git (push)
# upstream  https://github.com/facebookresearch/vjepa2.git (fetch)
# upstream  https://github.com/facebookresearch/vjepa2.git (push)
```

## 公式の最新を取り込む

### 1. mainを公式と同期

```bash
# 公式に新しいコミットがあるか確認
git fetch upstream
git log main..upstream/main --oneline

# 差分がある場合、mainを同期
git checkout main
git merge upstream/main
git push origin main
```

### 2. devに変更を取り込む

mainを同期した後、devにその変更を取り込みます。

**方法A: merge（履歴を保持、推奨）**
```bash
git checkout dev
git merge main
git push origin dev
```

**方法B: rebase（履歴をきれいに）**
```bash
git checkout dev
git rebase main
git push origin dev --force-with-lease
```

## 日常の開発

```bash
# devブランチで開発
git checkout dev

# 変更をコミット
git add <files>
git commit -m "your message"

# プッシュ
git push origin dev
```

## 注意事項

- `upstream` への push は行わない（読み取り専用として使用）
- devでの開発は公式リポジトリに影響しない
- 公式に貢献したい場合は、PRを送る必要がある
