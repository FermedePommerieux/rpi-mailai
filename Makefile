IMAGE ?= ghcr.io/<org>/rpi-mailai:latest

.PHONY: buildx-arm64 push-arm64 run

buildx-arm64:
	docker buildx build --platform linux/arm64 -f docker/Dockerfile -t $(IMAGE) .

push-arm64:
	docker buildx build --platform linux/arm64 -f docker/Dockerfile -t $(IMAGE) --push .

run:
	docker compose -f compose/docker-compose.yml up -d
