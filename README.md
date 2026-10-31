# misa-locomotion

脚ロボットの歩行の部品。articara（ロボットのモデル編集とシミュレーション）から 2026-10-31 に分けた。
履歴は articara の `crates/` の 10-29 以降（切り出しの最初のコミット 6fdaf28 から）を引き継ぐ。

| クレート | 中身 |
|---|---|
| `misa-mpc` | 単一剛体（重心まわり）のレンチ MPC。misa-wbc の QP／OCP の上に立つ |
| `misa-biped` | 二足の歩行器: 歩容・足跡・DCM・予見制御・接触計画・届く範囲・地形・計画（`Plan`）・押しの回復・設定とプリセット（`Config`、`kv`）・WBC の段（`TaskStack`）・歩行器本体（`walker::Walker`） |

依存は misarta・misa-wbc・nalgebra だけ（misa-biped は misa-mpc にも）。ライブラリは環境変数を読まず、印字もファイルの書き込みもしない（印字は `sink::LogSink`、診断は `sink::WalkerDiag` を通す）。
MPC の診断のトレースは `wrench_mpc::Config::trace_landing_time`／`trace_failures` で頼み、`solve_traced` の行の束で受け取る（歩行器はそれを `LogSink` の err に流す。ホストの名前は `MPC_LANDING_TIME_DEBUG`／`MPC_FS_DEBUG`）。

## 使い方

```toml
[dependencies]
misa-biped = { git = "https://github.com/takarakasai/misa-locomotion.git" }
# MPC だけなら
misa-mpc = { git = "https://github.com/takarakasai/misa-locomotion.git" }
```

```rust
use misa_biped::{config::Config, walker::WalkerInit};
// 設定: プリセット、または名前からの読み込み（ホストの環境変数なら |k| std::env::var(k).ok()）
let cfg = misa_biped::kv::config_from_kv(&profile, &|k| std::env::var(k).ok());
// 組み立て: WalkerInit::new(cfg, robot, setup, &mut sink) → .with_stack(stack) → Walker
// 毎 tick: walker.step(&observation, &mut sink, &mut diag) → StepOutput { command, record, state }
```

プリセット: `Config::g1_v619()`（Unitree G1）、`Config::kyo46rs_v2()`／`kyo46rs_v3()`（kyo46rs）。`tests/presets.rs` が `tests/fixtures/` の controls と項目ごとに照合する。

MuJoCo の試験の土台（機体のシーン、家族ベンチ）は articara の `examples/kyo46rs_walk/` にある。

## feature

- `misa-mpc/qp-hpipm`: HPIPM の構造化 OCP で MPC を解く（misa-wbc の `hpipm`、外部のソルバの準備が要る。misa-wbc の `native/README.md`）。

## 依存の misa-wbc

QP の診断・OCP・外部ソルバの feature を使うため、いまは misa-wbc のブランチ `articara-qp-ocp` を読む。main に合流したら `branch` を外す。

## テスト

```
cargo test --workspace
```
