//! The presets against the controls they came from, through the harness's
//! own key table (examples/kyo46rs_walk/config_env.rs, included as a module
//! so the table the walker runs with is the one under test). An environment
//! holding exactly a controls file's keys must give the preset, field for
//! field; and an empty one, the code defaults.
use misa_biped::config::Config;
use misa_biped::profile;
use std::collections::HashMap;

#[path = "../../../examples/kyo46rs_walk/config_env.rs"]
mod config_env;

/// A flat JSON object of string values (the shape of every controls file),
/// read without a JSON crate: misa-biped has none, and the files are written
/// by `json.dump` of a str -> str dict.
fn flat_json(rel: &str) -> HashMap<String, String> {
    let path = format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"));
    let s = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(!s.contains('\\'), "{path}: escapes are not supported");
    let t: Vec<&str> = s.split('"').collect();
    assert!(t.len() % 4 == 1 && t[0].trim() == "{" && t[t.len() - 1].trim() == "}", "{path}: not a flat object");
    let mut out = HashMap::new();
    for k in 0..t.len() / 4 {
        assert_eq!(t[4 * k + 2].trim(), ":", "{path}: a value that is not a string");
        if k > 0 {
            assert_eq!(t[4 * k].trim(), ",", "{path}: not a flat object");
        }
        out.insert(t[4 * k + 1].to_string(), t[4 * k + 3].to_string());
    }
    out
}

fn from_file(prof: &profile::Profile, rel: &str) -> Config {
    let kv = flat_json(rel);
    config_env::config_from_kv(prof, &|k| kv.get(k).cloned())
}

#[test]
fn g1_v619_is_its_controls() {
    let c = from_file(&profile::G1_23DOF, "log/claude_g1champion_20261029v619/controls.json");
    assert_eq!(c, Config::g1_v619());
}

#[test]
fn kyo46rs_v2_is_its_controls() {
    // ROBOT unset in the file: the harness's `by_name("")`.
    let c = from_file(&profile::by_name(""), "log/claude_g1rand_20260925/kyo_base_v2.json");
    assert_eq!(c, Config::kyo46rs_v2());
}

#[test]
fn no_keys_is_the_code_defaults() {
    for prof in [profile::KYO46RS, profile::G1_23DOF] {
        assert_eq!(config_env::config_from_kv(&prof, &|_| None), Config::code_defaults(&prof));
    }
}

#[test]
fn rough_mode_wins_over_its_knobs() {
    let kv: HashMap<&str, &str> = [("ROUGH_MODE", "1"), ("LIFT_H", "0.0512879"), ("SWING_Z_DROP", "0.7")].into();
    let c = config_env::config_from_kv(&profile::G1_23DOF, &|k| kv.get(k).map(|v| v.to_string()));
    assert_eq!(c.swing.lift_h, 0.08);
    assert!(c.swing.z_hold);
    assert_eq!(c.swing.z_drop, [0.50, 1.0]);
    assert_eq!(c.swing.xy_delay, 0.15);
    // On a known slope the apex defaults to 6 cm.
    let kv: HashMap<&str, &str> = [("ROUGH_MODE", "1"), ("TERRAIN_SLOPE", "3")].into();
    let c = config_env::config_from_kv(&profile::G1_23DOF, &|k| kv.get(k).map(|v| v.to_string()));
    assert_eq!(c.swing.lift_h, 0.06);
}
