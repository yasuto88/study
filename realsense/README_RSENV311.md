# rsenv311 — ローカルのみの Python 3.11 環境

このドキュメントは、`rsenv311` という名前の仮想環境をリポジトリ内に作成し、リモート（GitHub）にプッシュしないで管理する手順を説明します。

## 目的

- 開発や実験で使う Python 3.11 環境をリポジトリ配下に作成
- 環境ディレクトリは `.gitignore` に追加済みで、誤ってコミットされない

## 使い方（Windows PowerShell）

1. このリポジトリのルートに移動します。
2. スクリプトを実行して仮想環境を作成します。

```powershell
# リポジトリルートから
cd realsense
bash scripts/create_rsenv311.sh

3. 作成後、環境をアクティベートします。

```powershell
# 仮想環境をアクティブ化
& rsenv311\Scripts\Activate.ps1
```

4. 追加パッケージのインストール

- `librealsense/wrappers/python/requirements.txt` があれば自動的にそれを使います。
- 独自の依存がある場合は、ルートに `requirements-local.txt` を作成しておくと自動で読み込まれます。

## pyrealsense2 の注意点

`pyrealsense2` は OS/CPU アーキテクチャ依存のホイールが必要です。Windows向けの cp311 用の wheel がある場合は、次のようにインストールします:

```powershell
# 例: 手元に wheel がある場合
pip install C:\path\to\pyrealsense2‑win‑cp311‑...
```

wheel が無い場合は、ソースからビルドする必要があります。詳しくは Intel RealSense の公式ドキュメントを参照してください。

## セキュリティと Git 操作

- `rsenv311/` は `.gitignore` に追加済みです。コミット対象外になります。
- CI 用の再現性を高めるには `requirements-local.txt` に依存を固定しておくと良いです。

## トラブルシュート

- `python3.11` が見つからない場合は、システムにインストールされている Python 実行ファイル名（例: `python`）をスクリプト引数 `-Python` に渡してください。
- `pip install torch` は自動で CPU ビルドを取るように設定していますが、GPU を使う場合は PyTorch の公式サイトの指示に従ってください。
