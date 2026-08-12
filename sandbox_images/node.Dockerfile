FROM node:20-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && corepack enable

RUN mkdir -p /workspace && chown -R 1000:1000 /workspace
ENV npm_config_update_notifier=false CI=true
USER 1000:1000
WORKDIR /workspace
CMD ["npm", "test"]
