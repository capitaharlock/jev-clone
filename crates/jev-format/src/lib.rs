// JSONL I/O + V1 validation, mirroring `data/schema.py` (#T-rust-skel).
//
// V1 kinds: `choice` (≥2 options) + `boolean` (exactly 2 options, binary
// choice). `score` / `extract` / `multiselect` are V2-only and rejected.
use serde::{Deserialize, Serialize};

pub const V1_KINDS: [&str; 2] = ["choice", "boolean"];
const V2_KINDS: [&str; 3] = ["score", "extract", "multiselect"];
const VALID_SPLITS: [&str; 4] = ["train", "calibration", "test", "ood"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptionItem {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Question {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub options: Vec<OptionItem>,
    #[serde(default)]
    pub answer: Option<String>,
    #[serde(default)]
    pub teacher_conf: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Example {
    pub state: String,
    #[serde(default)]
    pub questions: Vec<Question>,
    #[serde(default = "default_split")]
    pub split: String,
}

fn default_split() -> String {
    "train".to_string()
}

pub fn validate(ex: &Example) -> Vec<String> {
    let mut errors = Vec::new();
    if ex.state.trim().is_empty() {
        errors.push("state must be non-empty".to_string());
    }
    if !VALID_SPLITS.contains(&ex.split.as_str()) {
        errors.push(format!("split must be one of {VALID_SPLITS:?}"));
    }
    if ex.questions.is_empty() {
        errors.push("at least one question required".to_string());
    }
    let mut seen_q = std::collections::HashSet::new();
    for q in &ex.questions {
        if !seen_q.insert(&q.id) {
            errors.push(format!("duplicate question id: {}", q.id));
        }
        if V2_KINDS.contains(&q.kind.as_str()) {
            errors.push(format!("question {}: kind {:?} is V2-only", q.id, q.kind));
        } else if !V1_KINDS.contains(&q.kind.as_str()) {
            errors.push(format!("question {}: unknown kind {:?}", q.id, q.kind));
        }
        let ids: Vec<&str> = q.options.iter().map(|o| o.id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        if unique.len() != ids.len() {
            errors.push(format!("question {}: duplicate option ids", q.id));
        }
        match q.kind.as_str() {
            "boolean" if q.options.len() != 2 => {
                errors.push(format!(
                    "question {}: boolean needs exactly 2 options",
                    q.id
                ));
            }
            "choice" if q.options.len() < 2 => {
                errors.push(format!("question {}: choice needs >= 2 options", q.id));
            }
            _ => {}
        }
        if let Some(a) = &q.answer {
            if a != "unknown" && !ids.contains(&a.as_str()) {
                errors.push(format!("question {}: answer not among options", q.id));
            }
        }
        if let Some(c) = q.teacher_conf {
            if !(0.0..=1.0).contains(&c) {
                errors.push(format!("question {}: teacher_conf outside [0,1]", q.id));
            }
        }
    }
    errors
}

pub fn is_valid(ex: &Example) -> bool {
    validate(ex).is_empty()
}

/// Parse one JSONL line; `Err` on bad JSON, `Ok` + errors on bad content.
pub fn parse_line(line: &str) -> Result<Example, String> {
    serde_json::from_str(line).map_err(|e| format!("bad JSONL: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Example {
        Example {
            state: "ctx".to_string(),
            split: "train".to_string(),
            questions: vec![Question {
                id: "q1".to_string(),
                kind: "choice".to_string(),
                options: vec![
                    OptionItem {
                        id: "a".to_string(),
                        text: "A".to_string(),
                    },
                    OptionItem {
                        id: "b".to_string(),
                        text: "B".to_string(),
                    },
                ],
                answer: Some("a".to_string()),
                teacher_conf: Some(0.9),
            }],
        }
    }

    #[test]
    fn valid_example_passes() {
        assert!(is_valid(&example()));
    }

    #[test]
    fn jsonl_round_trip() {
        let line = serde_json::to_string(&example()).unwrap();
        let back = parse_line(&line).unwrap();
        assert_eq!(back, example());
        assert!(is_valid(&back));
    }

    #[test]
    fn rejects_incomplete_records() {
        for mutate in [
            |e: &mut Example| e.state = "  ".to_string(),
            |e: &mut Example| e.questions.clear(),
            |e: &mut Example| e.questions[0].kind = "score".to_string(),
            |e: &mut Example| e.questions[0].options.truncate(1),
            |e: &mut Example| e.questions[0].answer = Some("zzz".to_string()),
            |e: &mut Example| e.questions[0].teacher_conf = Some(2.0),
            |e: &mut Example| e.split = "valid".to_string(),
        ] {
            let mut e = example();
            mutate(&mut e);
            assert!(!is_valid(&e), "should reject {e:?}");
        }
    }

    #[test]
    fn boolean_needs_exactly_two() {
        let mut e = example();
        e.questions[0].kind = "boolean".to_string();
        assert!(is_valid(&e));
        e.questions[0].options.pop();
        assert!(!is_valid(&e));
    }

    #[test]
    fn bad_json_is_an_error() {
        assert!(parse_line("{oops").is_err());
    }
}
