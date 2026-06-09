# Build the pwm demo CLI, then ship only the static binary on a slim runtime.
# Pure Rust, no C deps, no Python, no GPU: the image is small and the demo runs
# offline. rust-toolchain.toml is excluded by .dockerignore so the build uses the
# image's pinned 1.85 (the workspace MSRV) instead of resolving "stable".

FROM rust:1.85-slim-bookworm AS build
WORKDIR /src
# Copy only what cargo needs to build the binary (keeps layers cache-friendly).
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build -p pwm-testkit --bin pwm --release --locked

FROM debian:bookworm-slim
LABEL org.opencontainers.image.title="ProvableWorldModel demo (pwm)"
LABEL org.opencontainers.image.source="https://github.com/AbdelStark/ProvableWorldModel"
COPY --from=build /src/target/release/pwm /usr/local/bin/pwm
# Run the demo as an unprivileged user: it reads a proof bundle and prints, and
# needs no root capabilities. Least privilege by default.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin pwm
USER pwm
WORKDIR /home/pwm
# Default: play the whole demo. docker-compose overrides with prove / audit.
ENTRYPOINT ["pwm"]
CMD ["demo"]
