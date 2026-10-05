//! Agent-schema behaviours ported from upstream `_models.py` / `_loader.py`:
//! PowerFx `=` fields with `safe_mode`, the `template` block, list-form
//! properties, and nested-object schema normalization.

use std::collections::HashMap;

use agent_framework_declarative::{AgentSpec, DeclarativeLoader};
use serde_json::json;

fn env() -> impl Fn(&str) -> Option<String> {
    let vars: HashMap<&str, &str> = [("MODEL", "gpt-4.1"), ("ENV_MODE", "production")].into();
    move |k: &str| vars.get(k).map(|v| v.to_string())
}

#[test]
fn powerfx_fields_are_evaluated_and_env_requires_unsafe_mode() {
    let yaml = r#"
kind: Prompt
name: ="Assistant " & "One"
description: =Env.ENV_MODE = "production"
instructions: =Upper("be brief")
model:
  id: =Env.MODEL
  options:
    additionalProperties:
      keep: =not evaluated
metadata:
  note: =Upper("untouched")
"#;
    let safe = DeclarativeLoader::new().with_env(env());
    let spec = safe.load_agent_spec(yaml).unwrap();
    assert_eq!(spec.name.as_deref(), Some("Assistant One"));
    assert_eq!(spec.instructions.as_deref(), Some("BE BRIEF"));
    // Safe mode: Env is unbound, so the expression is kept verbatim.
    assert_eq!(
        spec.model.as_ref().unwrap().id.as_deref(),
        Some("=Env.MODEL")
    );
    assert_eq!(
        spec.description.as_deref(),
        Some("=Env.ENV_MODE = \"production\"")
    );
    assert_eq!(
        spec.metadata.as_ref().unwrap()["note"],
        json!("=Upper(\"untouched\")")
    );

    let unsafe_loader = DeclarativeLoader::new()
        .with_env(env())
        .with_safe_mode(false);
    let spec = unsafe_loader.load_agent_spec(yaml).unwrap();
    assert_eq!(spec.model.as_ref().unwrap().id.as_deref(), Some("gpt-4.1"));
    assert_eq!(spec.description.as_deref(), Some("true"));
    assert_eq!(
        spec.model.unwrap().options.unwrap().additional_properties["keep"],
        json!("=not evaluated")
    );
}

#[test]
fn invalid_expressions_are_kept_verbatim() {
    let spec = DeclarativeLoader::new()
        .load_agent_spec("kind: Prompt\nname: =1 +\ninstructions: =NoSuch(1)\n")
        .unwrap();
    assert_eq!(spec.name.as_deref(), Some("=1 +"));
    assert_eq!(spec.instructions.as_deref(), Some("=NoSuch(1)"));
}

#[test]
fn template_block_round_trips() {
    let yaml = r#"
kind: Prompt
name: T
template:
  format:
    kind: mustache
    strict: true
  parser:
    kind: prompty
    options:
      x: 1
"#;
    let spec = AgentSpec::from_yaml(yaml).unwrap();
    let template = spec.template.as_ref().unwrap();
    assert_eq!(
        template.format.as_ref().unwrap().kind.as_deref(),
        Some("mustache")
    );
    assert!(template.format.as_ref().unwrap().strict);
    assert_eq!(
        template.parser.as_ref().unwrap().kind.as_deref(),
        Some("prompty")
    );
    let again = AgentSpec::from_yaml(&spec.to_yaml().unwrap()).unwrap();
    assert_eq!(again, spec);
}

#[test]
fn list_form_properties_and_nested_object_schemas() {
    let yaml = r#"
kind: Prompt
name: S
outputSchema:
  properties:
    - name: answer
      kind: string
      required: true
    - name: details
      kind: object
      properties:
        - name: score
          kind: number
          enum: []
"#;
    let spec = AgentSpec::from_yaml(yaml).unwrap();
    let schema = spec.output_schema.unwrap().to_json_schema();
    assert_eq!(
        schema,
        json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "details": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                    "additionalProperties": false
                }
            },
            "required": ["answer"]
        })
    );
    let bad = "kind: Prompt\noutputSchema:\n  properties:\n    - kind: string\n";
    assert!(AgentSpec::from_yaml(bad)
        .unwrap_err()
        .to_string()
        .contains("'name'"));
}
