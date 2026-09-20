# jevclone cloud image (#T-cloud-api): multi-stage, no Python, no training data.
# Build: docker build -t jevclone:<ver> .
# Run:   docker run -p 8080:8080 jevclone:<ver>
ARG RUST=1.91
FROM rust:${RUST}-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --bin jevclone && \
    cp target/release/jevclone /jevclone

FROM debian:bookworm-slim
RUN useradd -r -u 10001 jev && apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /jevclone /usr/local/bin/jevclone
COPY sbom.json /sbom.json
USER jev
EXPOSE 8080
ENTRYPOINT ["jevclone", "serve", "--listen", "0.0.0.0:8080"]
