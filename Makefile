# Makefile
COMPOSE := docker compose
SERVICE := realsense

# Buildx ローカルキャッシュ（任意）
BUILD_CACHE_DIR := .buildx-cache

.PHONY: up down restart logs shell ps build rebuild clean prune cache-prune

## 起動（ビルド済みを使う）: make up
up:
	$(COMPOSE) up -d

## 初回 or 依存変更で再ビルド: make rebuild
rebuild:
	$(COMPOSE) build --pull --no-cache
	$(COMPOSE) up -d

## 通常ビルド（キャッシュあり）: make build
build:
	$(COMPOSE) build

## 停止: make down
down:
	$(COMPOSE) down

## 再起動: make restart
restart:
	$(COMPOSE) restart

## シェルに入る: make shell
shell:
	$(COMPOSE) exec $(SERVICE) bash -lc "source /opt/rsenv311/bin/activate; bash"

## ログ: make logs
logs:
	$(COMPOSE) logs -f --tail=200

## プロセス: make ps
ps:
	$(COMPOSE) ps

## コンテナ/ボリューム/ネットワークの掃除（注意）: make prune
prune:
	$(COMPOSE) down -v --remove-orphans
	docker system prune -f

## buildxキャッシュの掃除（任意）: make cache-prune
cache-prune:
	rm -rf $(BUILD_CACHE_DIR)
	docker builder prune -f

## BuildKitのキャッシュ活用ビルド（高速化したい場合）: make buildx
buildx:
	docker buildx build \
		--cache-from=type=local,src=$(BUILD_CACHE_DIR) \
		--cache-to=type=local,dest=$(BUILD_CACHE_DIR),mode=max \
		--tag realsense-dev:latest \
		--load .
