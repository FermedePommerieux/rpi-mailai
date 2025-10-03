IMAGE_REGISTRY ?= ghcr.io/mailai
IMAGE_NAME ?= rpi-mailai
IMAGE_TAG ?= latest
IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: build buildx-arm64 push-arm64 run

build:
\tdocker build --platform linux/arm64 -t $(IMAGE) -f docker/Dockerfile .

buildx-arm64:
\tdocker buildx build --platform linux/arm64 -t $(IMAGE) -f docker/Dockerfile .

push-arm64:
\tdocker buildx build --platform linux/arm64 -t $(IMAGE) -f docker/Dockerfile --push .

run:
\tdocker compose -f compose/docker-compose.yml up -d
