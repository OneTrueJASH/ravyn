# Contributing

RAVYN is developed inside [RAVYL](https://github.com/OneTrueJASH/ravyl), in
`crates/ravyn`, and mirrored here. Pull requests are welcome in either
repository; changes made here are applied there and synced back.

## Setup

```bash
git clone https://github.com/OneTrueJASH/ravyn
cd ravyn
cargo run --release --bin ravyn -- fetch-data   # the word list, into ~/.ravyn/data
```

You need Rust 1.88 or newer; nothing else is compiled. Tests that need the
word list skip cleanly without it.

## Before every push

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

CI runs the same and checks the MSRV.

## Measuring

Claims about speed or strength need numbers: `cargo bench --bench engine`
(criterion), or paired self-play games for strength.

## Commits

Small, focused commits with imperative subject lines.

## Reporting problems

Bugs and ideas: open an issue. Security problems: report them privately
(see [SECURITY.md](SECURITY.md)), never in a public issue.
