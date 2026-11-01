//! The presets against the controls they came from (copied into
//! tests/fixtures from articara's log/), through the key table hosts read
//! their knobs with (`misa_biped::kv`). An environment
//! holding exactly a controls file's keys must give the preset, field for
//! field; and an empty one, the code defaults.
use misa_biped::config::Config;
use misa_biped::profile;
use std::collections::HashMap;

use misa_biped::kv as config_env;

/// A flat JSON object of string values (the shape of every controls file),
/// read without a JSON crate: misa-biped has none, and the files are written
/// by `json.dump` of a str -> str dict.
fn flat_json(rel: &str) -> HashMap<String, String> {
    let path = format!("{}/tests/fixtures/{rel}", env!("CARGO_MANIFEST_DIR"));
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
    let c = from_file(&profile::G1_23DOF, "g1_v619.json");
    assert_eq!(c, Config::g1_v619());
}

#[test]
fn kyo46rs_v2_is_its_controls() {
    // ROBOT unset in the file: the harness's `by_name("")`.
    let c = from_file(&profile::by_name(""), "kyo46rs_v2.json");
    assert_eq!(c, Config::kyo46rs_v2());
}

#[test]
fn kyo46rs_v3_is_its_controls() {
    let c = from_file(&profile::by_name(""), "kyo46rs_v3.json");
    assert_eq!(c, Config::kyo46rs_v3());
    // v3's five stair-zone keys over v2 are the whole difference.
    let mut v2 = Config::kyo46rs_v2();
    v2.swing.stair_ff = false;
    v2.swing.absolute_td_desc = true;
    v2.stair_plan.align_stand_line = true;
    v2.descent.zone_crouch = 0.02;
    v2.swing.zone_rot_mode = 3;
    assert_eq!(c, v2);
    assert_ne!(c, Config::kyo46rs_v2());
}

#[test]
fn kyo46rs_v4_is_its_controls() {
    let c = from_file(&profile::by_name(""), "kyo46rs_v4.json");
    assert_eq!(c, Config::kyo46rs_v4());
    assert_ne!(c, Config::kyo46rs_v3());
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

/// `WALKER_PRESET` (S17): the preset is the base, and a key over it wins.
#[test]
fn walker_preset_is_the_base() {
    let one = |preset: &str| -> HashMap<String, String> { [("WALKER_PRESET".to_string(), preset.to_string())].into() };
    let kv = one("g1_v619");
    assert_eq!(config_env::config_from_kv(&profile::G1_23DOF, &|k| kv.get(k).cloned()), Config::g1_v619());
    let kv = one("kyo46rs_v2");
    assert_eq!(config_env::config_from_kv(&profile::by_name(""), &|k| kv.get(k).cloned()), Config::kyo46rs_v2());
    let kv = one("kyo46rs_v3");
    assert_eq!(config_env::config_from_kv(&profile::by_name(""), &|k| kv.get(k).cloned()), Config::kyo46rs_v3());
    let kv = one("kyo46rs_v4");
    assert_eq!(config_env::config_from_kv(&profile::by_name(""), &|k| kv.get(k).cloned()), Config::kyo46rs_v4());
    // Under its own controls the preset is unchanged (every key restates it).
    for (prof, rel, preset, want) in [
        (profile::G1_23DOF, "g1_v619.json", "g1_v619", Config::g1_v619()),
        (profile::by_name(""), "kyo46rs_v2.json", "kyo46rs_v2", Config::kyo46rs_v2()),
        (profile::by_name(""), "kyo46rs_v3.json", "kyo46rs_v3", Config::kyo46rs_v3()),
        (profile::by_name(""), "kyo46rs_v4.json", "kyo46rs_v4", Config::kyo46rs_v4()),
    ] {
        let mut kv = flat_json(rel);
        kv.insert("WALKER_PRESET".into(), preset.into());
        assert_eq!(config_env::config_from_kv(&prof, &|k| kv.get(k).cloned()), want, "{rel}");
    }
    // A key over the preset wins; the rest stays the preset's.
    let mut kv = one("g1_v619");
    kv.insert("KP_COM".into(), "123.5".into());
    kv.insert("ZMP_ROLL".into(), "0.01".into());
    kv.insert("WBC_STRATEGY".into(), "null_space".into());
    let c = config_env::config_from_kv(&profile::G1_23DOF, &|k| kv.get(k).cloned());
    let mut want = Config::g1_v619();
    want.gains.kp_com = 123.5;
    want.dcm.zmp_roll = Some(0.01);
    want.wbc.strategy = misa_wbc::HqpStrategy::NullSpace;
    assert_eq!(c, want);
}

#[test]
#[should_panic(expected = "WALKER_PRESET=g1_v619 is the G1 preset")]
fn walker_preset_for_another_robot_panics() {
    let kv: HashMap<&str, &str> = [("WALKER_PRESET", "g1_v619")].into();
    config_env::config_from_kv(&profile::KYO46RS, &|k| kv.get(k).map(|v| v.to_string()));
}
