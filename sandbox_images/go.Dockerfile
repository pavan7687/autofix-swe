FROM golang:1.22-bookworm

RUN mkdir -p /workspace /tmp/gocache \
 && chown -R 1000:1000 /workspace /tmp/gocache
ENV GOCACHE=/tmp/gocache GOFLAGS=-mod=mod CGO_ENABLED=0
USER 1000:1000
WORKDIR /workspace
CMD ["go", "test", "./..."]
