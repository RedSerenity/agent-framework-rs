//! Unit tests for the evaluation data model, checks and assertions. Modeled
//! on upstream's `tests/core/test_evaluation.py` plus the behaviors of
//! `_evaluation.py` that file does not cover directly.

#![allow(deprecated)]

use std::collections::HashMap;

use serde_json::{json, Value};

use super::checks::coerce_result;
use super::*;
use crate::tools::{hosted_web_search, FunctionTool};
use crate::types::{
    Content, DataContent, FunctionArguments, FunctionCallContent, FunctionResultContent,
};

fn tool(name: &str) -> ToolDefinition {
    FunctionTool::new(
        name,
        format!("{name} tool"),
        json!({"type": "object"}),
        |_| async { Ok(Value::Null) },
    )
    .into_definition()
}

fn call(name: &str, args: Option<FunctionArguments>) -> Message {
    Message::with_contents(
        "assistant",
        vec![Content::FunctionCall(FunctionCallContent::new(
            "c1", name, args,
        ))],
    )
}

fn obj_args(v: Value) -> Option<FunctionArguments> {
    let map: HashMap<String, Value> = v.as_object().unwrap().clone().into_iter().collect();
    Some(FunctionArguments::Object(map))
}

fn result(text: &str) -> Message {
    Message::with_contents(
        "tool",
        vec![Content::FunctionResult(FunctionResultContent::new(
            "c1",
            Some(json!(text)),
        ))],
    )
}

fn roles(messages: &[Message]) -> Vec<&str> {
    messages.iter().map(|m| m.role.as_str()).collect()
}

fn response(text: &str) -> AgentResponse {
    AgentResponse {
        messages: vec![Message::assistant(text)],
        ..AgentResponse::default()
    }
}

async fn run(check: &dyn EvalCheck, item: &EvalItem) -> CheckResult {
    check.check(item).await.unwrap()
}

// ---------------------------------------------------------------------------
// to_eval_item (upstream TestToEvalItem)
// ---------------------------------------------------------------------------

#[test]
fn string_query_builds_user_then_response() {
    let item =
        EvalItem::from_agent_response("What's the weather?", &response("The weather is sunny."));
    assert_eq!(item.query(), "What's the weather?");
    assert_eq!(item.response(), "The weather is sunny.");
    assert_eq!(roles(&item.conversation), ["user", "assistant"]);
    assert!(item.tools.is_none());
}

#[test]
fn message_query_keeps_all_input_messages() {
    let item = EvalItem::from_agent_response(
        vec![Message::system("Be helpful."), Message::user("Hello")],
        &response("Hi there!"),
    );
    assert_eq!(item.query(), "Hello");
    assert_eq!(item.conversation.len(), 3);
}

#[test]
fn context_is_carried() {
    let item = to_eval_item(
        "Question?".into(),
        &response("Answer."),
        None,
        None,
        Some("Some reference document.".into()),
    );
    assert_eq!(item.context.as_deref(), Some("Some reference document."));
}

#[test]
fn explicit_tools_are_kept_and_hosted_tools_dropped() {
    let item = to_eval_item(
        "Find info".into(),
        &response("Found it."),
        None,
        Some(vec![tool("search"), hosted_web_search()]),
        None,
    );
    let tools = item.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "search");
}

struct ToolAgent(Vec<ToolDefinition>);

#[async_trait]
impl SupportsAgentRun for ToolAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut crate::session::AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(response("ok"))
    }
    fn id(&self) -> &str {
        "tool-agent"
    }
    fn default_tools(&self) -> Vec<ToolDefinition> {
        self.0.clone()
    }
}

#[test]
fn agent_tools_are_used_when_no_explicit_tools() {
    let agent = ToolAgent(vec![tool("calculate"), tool("search")]);
    let item = to_eval_item(
        "Research this".into(),
        &response("Done"),
        Some(&agent),
        None,
        None,
    );
    let names: Vec<String> = item.tools.unwrap().into_iter().map(|t| t.name).collect();
    assert_eq!(names, ["calculate", "search"]);
}

#[test]
fn explicit_tools_override_agent_tools() {
    let agent = ToolAgent(vec![tool("agent_tool")]);
    let item = to_eval_item(
        "Test".into(),
        &response("Done"),
        Some(&agent),
        Some(vec![tool("explicit_tool")]),
        None,
    );
    let names: Vec<String> = item.tools.unwrap().into_iter().map(|t| t.name).collect();
    assert_eq!(names, ["explicit_tool"]);
    // An empty explicit list falls back to the agent's, as upstream's `if tools:`.
    let item = to_eval_item(
        "Test".into(),
        &response("Done"),
        Some(&agent),
        Some(vec![]),
        None,
    );
    assert_eq!(item.tools.unwrap()[0].name, "agent_tool");
}

#[test]
fn agent_builder_exposes_default_tools() {
    struct Nop;
    #[async_trait]
    impl crate::client::ChatClient for Nop {
        async fn get_response(
            &self,
            _m: Vec<Message>,
            _o: crate::types::ChatOptions,
        ) -> Result<crate::types::ChatResponse> {
            Ok(crate::types::ChatResponse::from_text("x"))
        }
        async fn get_streaming_response(
            &self,
            _m: Vec<Message>,
            _o: crate::types::ChatOptions,
        ) -> Result<crate::client::ChatStream> {
            unimplemented!()
        }
    }
    let agent = crate::agent::Agent::builder(Nop).tool(tool("t")).build();
    let item = EvalItem::from_agent_response("q", &response("r")).with_agent_tools(&agent);
    assert_eq!(item.tools.unwrap()[0].name, "t");
}

// ---------------------------------------------------------------------------
// Splitting (upstream TestEvalItemSplitting)
// ---------------------------------------------------------------------------

#[test]
fn split_messages_format() {
    let item = EvalItem::from_agent_response("Q", &response("Answer")).with_tools([tool("test")]);
    let (q, r) = item.split_messages();
    assert_eq!(roles(&q), ["user"]);
    assert_eq!(roles(&r), ["assistant"]);
    assert_eq!(item.tools.unwrap().len(), 1);
}

fn multiturn() -> Vec<Message> {
    vec![
        Message::user("What's the weather?"),
        Message::assistant("It's sunny in Seattle."),
        Message::user("And tomorrow?"),
        call("get_forecast", None),
        result("Rain expected"),
        Message::assistant("Rain is expected tomorrow."),
    ]
}

#[test]
fn multiturn_preserves_interleaving() {
    let (q, r) = EvalItem::new(multiturn()).split_messages();
    assert_eq!(roles(&q), ["user", "assistant", "user"]);
    assert_eq!(roles(&r), ["assistant", "tool", "assistant"]);
}

#[test]
fn full_split() {
    let conversation = vec![
        Message::user("What's the weather?"),
        Message::assistant("It's 62°F in Seattle."),
        Message::user("And tomorrow?"),
        Message::assistant("Rain is expected tomorrow."),
    ];
    let (q, r) = EvalItem::new(conversation).split_messages_with(&ConversationSplit::Full);
    assert_eq!(
        q.iter().map(Message::text).collect::<Vec<_>>(),
        ["What's the weather?"]
    );
    assert_eq!(roles(&r), ["assistant", "user", "assistant"]);
}

#[test]
fn full_split_includes_system_message() {
    let conversation = vec![
        Message::system("You are a weather assistant."),
        Message::user("What's the weather?"),
        Message::assistant("It's sunny."),
    ];
    let (q, r) = EvalItem::new(conversation).split_messages_with(&ConversationSplit::Full);
    assert_eq!(roles(&q), ["system", "user"]);
    assert_eq!(roles(&r), ["assistant"]);
}

#[test]
fn full_split_puts_tool_interactions_in_response() {
    let conversation = vec![
        Message::user("What's the weather?"),
        call("get_weather", None),
        result("62°F"),
        Message::assistant("It's 62°F."),
        Message::user("Thanks!"),
        Message::assistant("You're welcome!"),
    ];
    let (q, r) = EvalItem::new(conversation).split_messages_with(&ConversationSplit::Full);
    assert_eq!(q.len(), 1);
    assert_eq!(r.len(), 5);
}

#[test]
fn no_user_message_puts_everything_in_the_response() {
    let conversation = vec![Message::assistant("Hello")];
    for split in [ConversationSplit::LastTurn, ConversationSplit::Full] {
        let (q, r) = split.split(&conversation);
        assert!(q.is_empty());
        assert_eq!(r.len(), 1);
    }
}

#[test]
fn last_turn_is_default() {
    let item = EvalItem::new(vec![
        Message::user("Hello"),
        Message::assistant("Hi there"),
        Message::user("Bye"),
        Message::assistant("Goodbye"),
    ]);
    assert_eq!(
        item.split_messages(),
        item.split_messages_with(&ConversationSplit::LastTurn)
    );
    assert!(item.split_strategy.is_none());
    // The query joins every user message on the query side.
    assert_eq!(item.query(), "Hello Bye");
    assert_eq!(item.response(), "Goodbye");
}

#[test]
fn per_turn_items() {
    let conversation = vec![
        Message::user("What's the weather?"),
        Message::assistant("It's 62°F."),
        Message::user("And tomorrow?"),
        Message::assistant("Rain expected."),
    ];
    let items = EvalItem::per_turn_items(&conversation, None, None);
    assert_eq!(items.len(), 2);
    assert_eq!(
        (items[0].query(), items[0].response()),
        ("What's the weather?".to_string(), "It's 62°F.".to_string())
    );
    assert_eq!(
        (items[1].query(), items[1].response()),
        (
            "What's the weather? And tomorrow?".to_string(),
            "Rain expected.".to_string()
        )
    );
    assert_eq!(
        items
            .iter()
            .map(|i| i.conversation.len())
            .collect::<Vec<_>>(),
        [2, 4]
    );
}

#[test]
fn per_turn_items_preserve_tools_and_context() {
    let conversation = vec![
        Message::user("Check weather"),
        call("get_weather", None),
        result("sunny"),
        Message::assistant("It's sunny."),
        Message::user("Thanks"),
        Message::assistant("You're welcome!"),
    ];
    let items = EvalItem::per_turn_items(
        &conversation,
        Some(vec![tool("get_weather")]),
        Some("ctx".into()),
    );
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].tools.as_ref().unwrap()[0].name, "get_weather");
    assert_eq!(items[1].context.as_deref(), Some("ctx"));
    assert_eq!(items[0].response(), "It's sunny.");
    assert_eq!(items[1].response(), "You're welcome!");
}

#[test]
fn per_turn_items_edge_cases() {
    assert!(EvalItem::per_turn_items(&[Message::assistant("Hello")], None, None).is_empty());
    let items = EvalItem::per_turn_items(
        &[Message::user("Hi"), Message::assistant("Hello!")],
        None,
        None,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(
        (items[0].query(), items[0].response()),
        ("Hi".into(), "Hello!".into())
    );
}

fn split_before_memory(messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
    for (i, m) in messages.iter().enumerate() {
        if m.contents
            .iter()
            .any(|c| matches!(c, Content::FunctionCall(fc) if fc.name == "retrieve_memory"))
        {
            return (messages[..i].to_vec(), messages[i..].to_vec());
        }
    }
    split_last_turn(messages)
}

#[test]
fn custom_splitter_callable_and_fallback() {
    let conversation = vec![
        Message::user("Remember my name is Alice"),
        Message::assistant("Got it, Alice!"),
        Message::user("What's the capital of France?"),
        call("retrieve_memory", None),
        result("User name: Alice"),
        Message::assistant("The capital of France is Paris, Alice!"),
    ];
    let (q, r) = EvalItem::new(conversation).split_messages_with(&split_before_memory);
    assert_eq!(q.len(), 3);
    assert_eq!(q.last().unwrap().role.as_str(), "user");
    assert_eq!(r.len(), 3);
    assert_eq!(r[0].role.as_str(), "assistant");

    let (q, r) = EvalItem::new(vec![
        Message::user("Hello"),
        Message::assistant("Hi there!"),
    ])
    .split_messages_with(&split_before_memory);
    assert_eq!(roles(&q), ["user"]);
    assert_eq!(roles(&r), ["assistant"]);
}

#[test]
fn custom_splitter_closure() {
    let conversation = vec![
        Message::user("A"),
        Message::assistant("B"),
        Message::user("C"),
        Message::assistant("D"),
    ];
    let split = |m: &[Message]| (m[..2].to_vec(), m[2..].to_vec());
    let (q, r) = EvalItem::new(conversation).split_messages_with(&split);
    assert_eq!((q.len(), r.len()), (2, 2));
}

#[test]
fn item_split_strategy_is_default_and_explicit_overrides_it() {
    let conversation = vec![
        Message::user("First"),
        Message::assistant("Response 1"),
        Message::user("Second"),
        Message::assistant("Response 2"),
    ];
    let item = EvalItem::new(conversation).with_split_strategy(ConversationSplit::Full);
    let (q, r) = item.split_messages();
    assert_eq!(q.iter().map(Message::text).collect::<Vec<_>>(), ["First"]);
    assert_eq!(r.len(), 3);
    // query/response follow the item strategy too.
    assert_eq!(item.query(), "First");
    assert_eq!(item.response(), "Response 1 Response 2");

    let (q, r) = item.split_messages_with(&ConversationSplit::LastTurn);
    assert_eq!(q.len(), 3);
    assert_eq!(q.last().unwrap().text(), "Second");
    assert_eq!(r.len(), 1);
}

#[test]
fn conversation_split_strings() {
    assert_eq!(ConversationSplit::LastTurn.as_str(), "last_turn");
    assert_eq!(ConversationSplit::Full.as_str(), "full");
    assert_eq!(
        serde_json::to_value(ConversationSplit::Full).unwrap(),
        json!("full")
    );
    assert_eq!(ConversationSplit::default(), ConversationSplit::LastTurn);
}

// ---------------------------------------------------------------------------
// Legacy converter (upstream TestAgentEvalConverterCompatibility)
// ---------------------------------------------------------------------------

#[test]
fn legacy_convert_messages_preserves_wire_format() {
    let messages = vec![
        Message::user("What's the weather?"),
        call("get_weather", obj_args(json!({"city": "Seattle"}))),
    ];
    let converted = AgentEvalConverter::convert_messages(&messages);
    assert_eq!(
        converted,
        vec![
            json!({"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]}),
            json!({"role": "assistant", "content": [{
                "type": "tool_call", "tool_call_id": "c1", "name": "get_weather",
                "arguments": {"city": "Seattle"}
            }]}),
        ]
    );
}

#[test]
fn legacy_convert_message_edge_cases() {
    // Unparseable raw arguments, tool results, images, and empty messages.
    let raw = call("f", Some(FunctionArguments::Raw("{oops".into())));
    assert_eq!(
        AgentEvalConverter::convert_message(&raw)[0]["content"][0]["arguments"],
        json!({"_raw_arguments": "[unparseable]"})
    );
    let tool_msg = Message::with_contents(
        "tool",
        vec![
            Content::FunctionResult(FunctionResultContent::new("a", Some(json!("{\"t\": 1}")))),
            Content::FunctionResult(FunctionResultContent::new("b", Some(json!("plain")))),
        ],
    );
    assert_eq!(
        AgentEvalConverter::convert_message(&tool_msg),
        vec![
            json!({"role": "tool", "tool_call_id": "a", "content": [{"type": "tool_result", "tool_result": {"t": 1}}]}),
            json!({"role": "tool", "tool_call_id": "b", "content": [{"type": "tool_result", "tool_result": "plain"}]}),
        ]
    );
    let image = Message::with_contents(
        "user",
        vec![Content::Data(DataContent::from_bytes(b"x", "image/png"))],
    );
    let converted = AgentEvalConverter::convert_message(&image);
    assert_eq!(converted[0]["content"][0]["type"], "input_image");
    assert_eq!(converted[0]["content"][0]["detail"], "auto");
    let empty = Message::with_contents("assistant", vec![]);
    assert_eq!(
        AgentEvalConverter::convert_message(&empty),
        vec![json!({"role": "assistant", "content": [{"type": "text", "text": ""}]})]
    );
}

#[test]
fn legacy_extract_tools_and_to_eval_item() {
    let agent = ToolAgent(vec![tool("a"), tool("a"), hosted_web_search(), tool("b")]);
    let tools = AgentEvalConverter::extract_tools(&agent);
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "a");
    assert_eq!(tools[0]["parameters"], json!({"type": "object"}));

    let item = AgentEvalConverter::to_eval_item("Weather?", &response("Sunny."), None, None, None);
    assert_eq!(item.query(), "Weather?");
    assert_eq!(item.response(), "Sunny.");
}

// ---------------------------------------------------------------------------
// Built-in checks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn keyword_check_matches_case_insensitively_by_default() {
    let item = EvalItem::from_agent_response("q", &response("The Weather is warm"));
    let r = run(&keyword_check(["weather", "warm"]), &item).await;
    assert!(r.passed);
    assert_eq!(r.reason, "All keywords found");
    assert_eq!(r.check_name, "keyword_check");

    let r = run(&keyword_check(["weather", "Rain"]), &item).await;
    assert!(!r.passed);
    assert_eq!(r.reason, "Missing keywords: ['Rain']");

    let r = run(&keyword_check(["weather"]).case_sensitive(true), &item).await;
    assert!(!r.passed);
    assert_eq!(r.reason, "Missing keywords: ['weather']");
}

fn tool_conversation() -> EvalItem {
    EvalItem::new(vec![
        Message::user("q"),
        call("b_tool", obj_args(json!({"location": "NYC", "units": "f"}))),
        call("a_tool", Some(FunctionArguments::Raw("{\"n\": 1}".into()))),
        Message::assistant("done"),
    ])
}

#[tokio::test]
async fn tool_called_check_all_mode() {
    let item = tool_conversation();
    let r = run(&tool_called_check(["a_tool", "b_tool"]), &item).await;
    assert!(r.passed);
    assert_eq!(r.reason, "All expected tools called: ['a_tool', 'b_tool']");
    assert_eq!(r.check_name, "tool_called");

    let r = run(&tool_called_check(["b_tool", "c_tool"]), &item).await;
    assert!(!r.passed);
    assert_eq!(
        r.reason,
        "Expected tools not called: ['c_tool'] (called: ['a_tool', 'b_tool'])"
    );

    // Early return reports the calls seen so far.
    let r = run(&tool_called_check(["b_tool"]), &item).await;
    assert_eq!(r.reason, "All expected tools called: ['b_tool']");

    // No calls and no expectations: vacuous pass.
    let empty = EvalItem::new(vec![Message::user("q")]);
    let r = run(&tool_called_check(Vec::<String>::new()), &empty).await;
    assert!(r.passed);
    assert_eq!(r.reason, "All expected tools called: []");
}

#[tokio::test]
async fn tool_called_check_any_mode() {
    let item = tool_conversation();
    let r = run(
        &tool_called_check(["zzz", "a_tool"]).mode(ToolCalledMode::Any),
        &item,
    )
    .await;
    assert!(r.passed);
    assert_eq!(r.reason, "Expected tool found: ['a_tool']");

    let r = run(
        &tool_called_check(["x", "y"]).mode(ToolCalledMode::Any),
        &item,
    )
    .await;
    assert!(!r.passed);
    assert_eq!(
        r.reason,
        "None of expected tools called: ['x', 'y'] (called: ['a_tool', 'b_tool'])"
    );
}

#[tokio::test]
async fn tool_calls_present_check() {
    let item = tool_conversation();
    let r = run(&tool_calls_present, &item).await;
    assert!(r.passed);
    assert_eq!(r.reason, "No expected tool calls specified.");

    let item = tool_conversation().with_expected_tool_calls([
        ExpectedToolCall::new("a_tool"),
        ExpectedToolCall::new("b_tool"),
    ]);
    let r = run(&tool_calls_present, &item).await;
    assert!(r.passed);
    assert_eq!(
        r.reason,
        "All expected tools called: ['a_tool', 'b_tool'] (called: ['a_tool', 'b_tool'])"
    );
    assert_eq!(r.check_name, "tool_calls_present");

    let item = tool_conversation().with_expected_tool_calls([ExpectedToolCall::new("nope")]);
    let r = run(&tool_calls_present, &item).await;
    assert!(!r.passed);
    assert_eq!(
        r.reason,
        "Missing tool calls: ['nope'] (called: ['a_tool', 'b_tool'])"
    );
}

#[tokio::test]
async fn tool_call_args_match_check() {
    let item = tool_conversation().with_expected_tool_calls([
        ExpectedToolCall::new("b_tool").with_arguments([("location", json!("NYC"))]),
        ExpectedToolCall::new("a_tool").with_arguments([("n", json!(1.0))]),
        ExpectedToolCall::new("b_tool"),
    ]);
    let r = run(&tool_call_args_match, &item).await;
    assert!(r.passed, "{}", r.reason);
    assert_eq!(
        r.reason,
        "Tool call args match: 3/3\n  b_tool: args match\n  a_tool: args match\n  b_tool: called (args not checked)"
    );

    let item = tool_conversation().with_expected_tool_calls([
        ExpectedToolCall::new("b_tool").with_arguments([("location", json!("LA"))]),
        ExpectedToolCall::new("missing"),
    ]);
    let r = run(&tool_call_args_match, &item).await;
    assert!(!r.passed);
    assert_eq!(
        r.reason,
        "Tool call args match: 0/2\n  b_tool: args mismatch (actual: [{'location': 'NYC', 'units': 'f'}])\n  missing: not called"
    );
    assert_eq!(r.check_name, "tool_call_args_match");
}

#[tokio::test]
async fn tool_call_args_parse_rules() {
    // Blank raw args are {}, unparseable/non-object args are None (never match).
    let item = EvalItem::new(vec![
        Message::user("q"),
        call("blank", Some(FunctionArguments::Raw("  ".into()))),
        call("bad", Some(FunctionArguments::Raw("[1, 2]".into()))),
    ])
    .with_expected_tool_calls([
        ExpectedToolCall::new("blank").with_arguments(Vec::<(String, Value)>::new()),
        ExpectedToolCall::new("bad").with_arguments([("x", json!(1))]),
    ]);
    let r = run(&tool_call_args_match, &item).await;
    assert!(!r.passed);
    assert!(r.reason.contains("blank: args match"));
    assert!(r.reason.contains("bad: args mismatch (actual: [None])"));
}

// ---------------------------------------------------------------------------
// Function evaluators and coercion
// ---------------------------------------------------------------------------

#[test]
fn coerce_bool_and_numbers() {
    let r = coerce_result(EvalOutcome::Bool(true), "c").unwrap();
    assert_eq!(
        (r.passed, r.reason.as_str(), r.check_name.as_str()),
        (true, "passed", "c")
    );
    assert_eq!(
        coerce_result(EvalOutcome::Bool(false), "c").unwrap().reason,
        "failed"
    );
    let r = coerce_result(EvalOutcome::Score(0.5), "c").unwrap();
    assert!(r.passed);
    assert_eq!(r.reason, "score=0.500");
    assert!(!coerce_result(EvalOutcome::Score(0.49), "c").unwrap().passed);
}

fn map(v: Value) -> EvalOutcome {
    EvalOutcome::Map(v.as_object().unwrap().clone())
}

#[test]
fn coerce_dicts() {
    let r = coerce_result(map(json!({"score": 0.7})), "c").unwrap();
    assert!(r.passed);
    assert_eq!(r.reason, "score=0.700");
    let r = coerce_result(map(json!({"score": 0.7, "threshold": 0.8})), "c").unwrap();
    assert!(!r.passed);
    let r = coerce_result(
        map(json!({"score": 0.1, "passed": true, "reason": "ok"})),
        "c",
    )
    .unwrap();
    assert!(r.passed);
    assert_eq!(r.reason, "ok");
    let r = coerce_result(map(json!({"score": "0.9"})), "c").unwrap();
    assert!(r.passed);
    let r = coerce_result(map(json!({"passed": 0})), "c").unwrap();
    assert!(!r.passed);
    assert_eq!(r.reason, "failed");
    let r = coerce_result(map(json!({"passed": true, "reason": 3})), "c").unwrap();
    assert_eq!(r.reason, "3");

    let err = coerce_result(map(json!({"score": "high"})), "c").unwrap_err();
    assert!(err
        .to_string()
        .contains("non-numeric 'score' value: 'high'"));
    let err = coerce_result(map(json!({"passed": "yes"})), "c").unwrap_err();
    assert!(err
        .to_string()
        .contains("non-boolean 'passed' value: 'yes'"));
    let err = coerce_result(map(json!({"other": 1})), "c").unwrap_err();
    assert!(err.to_string().contains("unsupported type dict"));
    let err = coerce_result(EvalOutcome::Unsupported("str".into()), "c").unwrap_err();
    assert_eq!(
        err.to_string(),
        "Function evaluator 'c' returned unsupported type str. Expected bool, float, dict, or CheckResult."
    );
}

#[test]
fn values_convert_to_outcomes() {
    assert_eq!(
        json!(true).into_eval_outcome().unwrap(),
        EvalOutcome::Bool(true)
    );
    assert_eq!(
        json!(1).into_eval_outcome().unwrap(),
        EvalOutcome::Score(1.0)
    );
    assert_eq!(
        json!("x").into_eval_outcome().unwrap(),
        EvalOutcome::Unsupported("str".into())
    );
    assert!(Err::<bool, _>(Error::other("boom"))
        .into_eval_outcome()
        .is_err());
}

#[tokio::test]
async fn function_evaluators_receive_every_field() {
    let item = EvalItem::from_agent_response("What's the weather?", &response("Sunny"))
        .with_expected_output("Sunny")
        .with_context("ctx")
        .with_tools([tool("get_weather")])
        .with_expected_tool_calls([ExpectedToolCall::new("get_weather")]);
    let check = evaluator("exact", |f: EvalFields| {
        assert_eq!(f.query, "What's the weather?");
        assert_eq!(f.conversation.len(), 2);
        assert_eq!(f.context.as_deref(), Some("ctx"));
        assert_eq!(f.tools.as_ref().unwrap().len(), 1);
        assert_eq!(f.expected_tool_calls.len(), 1);
        f.response == f.expected_output
    });
    assert_eq!(check.name(), "exact");
    let r = run(&check, &item).await;
    assert_eq!((r.passed, r.check_name.as_str()), (true, "exact"));

    // Unset optional fields default to empty values.
    let bare = EvalItem::new(vec![Message::user("q")]);
    let check = evaluator("defaults", |f: EvalFields| {
        f.expected_output.is_empty() && f.expected_tool_calls.is_empty() && f.tools.is_none()
    });
    assert!(run(&check, &bare).await.passed);
}

#[tokio::test]
async fn async_evaluators_and_errors() {
    let item = EvalItem::from_agent_response("q", &response("a"));
    let judge = async_evaluator("judge", |f: EvalFields| async move {
        json!({"score": if f.response == "a" { 0.9 } else { 0.1 }, "reason": "judged"})
    });
    let r = run(&judge, &item).await;
    assert!(r.passed);
    assert_eq!(r.reason, "judged");

    let failing = evaluator("boom", |_f: EvalFields| -> Result<bool> {
        Err(Error::other("nope"))
    });
    assert!(failing.check(&item).await.is_err());
    let bad = evaluator("bad", |_f: EvalFields| json!(["list"]));
    let err = bad.check(&item).await.unwrap_err();
    assert!(err.to_string().contains("unsupported type list"));
}

// ---------------------------------------------------------------------------
// LocalEvaluator
// ---------------------------------------------------------------------------

#[tokio::test]
async fn local_evaluator_aggregates_per_item_and_per_check() {
    let local = LocalEvaluator::new()
        .with_check(keyword_check(["sunny"]))
        .with_check(evaluator("short", |f: EvalFields| f.response.len() < 10));
    let items = vec![
        EvalItem::from_agent_response("q1", &response("sunny")),
        EvalItem::from_agent_response("q2", &response("rain and more rain")),
    ];
    let results = local.evaluate(&items, "My Eval").await.unwrap();
    assert_eq!(results.provider, "Local");
    assert_eq!(results.eval_id, "local");
    assert_eq!(results.run_id, "My Eval");
    assert_eq!(results.status, EvalRunStatus::Completed);
    assert_eq!(
        (results.passed(), results.failed(), results.total()),
        (1, 1, 2)
    );
    assert_eq!(
        results.per_evaluator["keyword_check"],
        ResultCounts::new(1, 1, 0)
    );
    assert_eq!(results.per_evaluator["short"], ResultCounts::new(1, 1, 0));
    assert_eq!(
        results.error.as_deref(),
        Some("keyword_check: Missing keywords: ['sunny']; short: failed")
    );
    let item0 = &results.items[0];
    assert_eq!(item0.item_id, "0");
    assert!(item0.is_passed());
    assert_eq!(item0.input_text.as_deref(), Some("q1"));
    assert_eq!(item0.output_text.as_deref(), Some("sunny"));
    assert_eq!(item0.scores[0].score, 1.0);
    assert_eq!(
        item0.scores[0].sample,
        Some(json!({"reason": "All keywords found"}))
    );
    assert!(results.items[1].is_failed());
    assert_eq!(results.items[1].scores[0].score, 0.0);
    assert!(!results.all_passed());
}

#[tokio::test]
async fn local_evaluator_without_checks_fails_items() {
    let results = LocalEvaluator::new()
        .evaluate(&[EvalItem::new(vec![Message::user("q")])], "e")
        .await
        .unwrap();
    assert_eq!((results.passed(), results.failed()), (0, 1));
    assert!(results.error.is_none());
    assert!(results.per_evaluator.is_empty());
}

#[tokio::test]
async fn closure_checks_and_check_errors() {
    let local = LocalEvaluator::new().with_check(|item: &EvalItem| {
        CheckResult::new(!item.response().is_empty(), "", "non_empty")
    });
    let results = local
        .evaluate(&[EvalItem::from_agent_response("q", &response("a"))], "e")
        .await
        .unwrap();
    assert!(results.all_passed());
    // Empty reasons yield no sample.
    assert!(results.items[0].scores[0].sample.is_none());

    let failing = LocalEvaluator::new()
        .with_check(evaluator("boom", |_f: EvalFields| -> Result<bool> {
            Err(Error::other("boom"))
        }));
    assert!(failing
        .evaluate(&[EvalItem::new(vec![Message::user("q")])], "e")
        .await
        .is_err());
}

// ---------------------------------------------------------------------------
// EvalResults
// ---------------------------------------------------------------------------

fn item_result(id: &str, status: &str, scores: Vec<EvalScoreResult>) -> EvalItemResult {
    let mut r = EvalItemResult::new(id, status);
    r.scores = scores;
    r
}

#[test]
fn item_status_parsing() {
    assert!(EvalItemResult::new("1", "errored").is_error());
    assert!(EvalItemResult::new("1", "error").is_error());
    assert!(EvalItemResult::new("1", "pass").is_passed());
    assert!(EvalItemResult::new("1", "fail").is_failed());
    let other = EvalItemResult::new("1", "queued");
    assert!(!other.is_error() && !other.is_passed() && !other.is_failed());
    assert_eq!(other.status.to_string(), "queued");
    assert_eq!(
        serde_json::to_value(EvalItemStatus::Pass).unwrap(),
        json!("pass")
    );
    assert_eq!(EvalRunStatus::from("timeout"), EvalRunStatus::Timeout);
    assert_eq!(
        EvalRunStatus::from("in_progress").to_string(),
        "in_progress"
    );
}

#[test]
fn all_passed_rules() {
    let ok = EvalResults::new("p").with_result_counts(ResultCounts::new(2, 0, 0));
    assert!(ok.all_passed());
    assert!(!EvalResults::new("p").all_passed(), "no counts, no subs");
    assert!(!EvalResults::new("p")
        .with_result_counts(ResultCounts::new(0, 0, 0))
        .all_passed());
    assert!(!EvalResults::new("p")
        .with_result_counts(ResultCounts::new(2, 0, 1))
        .all_passed());
    assert!(!ok.clone().with_status("failed").all_passed());

    // A parent without counts defers to its sub-results.
    let parent = EvalResults::new("p").with_sub_result("a", ok.clone());
    assert!(parent.all_passed());
    let bad_sub = EvalResults::new("p").with_result_counts(ResultCounts::new(1, 1, 0));
    assert!(!parent.clone().with_sub_result("b", bad_sub).all_passed());
    // A parent with counts must pass itself too.
    assert!(!EvalResults::new("p")
        .with_result_counts(ResultCounts::new(0, 1, 0))
        .with_sub_result("a", ok)
        .all_passed());
}

#[test]
fn raise_for_status_message() {
    let ok = EvalResults::new("p").with_result_counts(ResultCounts::new(1, 0, 0));
    assert!(ok.raise_for_status(None).is_ok());

    let mut failing = EvalResults::new("p")
        .with_run_id("run1")
        .with_result_counts(ResultCounts::new(1, 2, 3))
        .with_report_url("https://portal/x")
        .with_error("boom")
        .with_sub_result(
            "agent_b",
            EvalResults::new("p").with_result_counts(ResultCounts::new(0, 1, 0)),
        )
        .with_sub_result("agent_a", ok.clone());
    let mut errored = EvalItemResult::new("i1", "error");
    errored.error_code = Some("QueryExtractionError".into());
    failing.items = vec![
        errored,
        EvalItemResult::new("i2", "errored"),
        EvalItemResult::new("i3", "fail"),
    ];
    let err = failing.raise_for_status(None).unwrap_err();
    assert!(matches!(err, Error::EvalNotPassed(_)));
    let Error::EvalNotPassed(msg) = err else {
        unreachable!()
    };
    assert_eq!(
        msg,
        "Eval run run1 completed: 1 passed, 2 failed. 3 errored. See https://portal/x for details. \
         Error: boom Failed: agent_b. Errored items: i1: QueryExtractionError, i2: unknown."
    );

    let Error::EvalNotPassed(msg) = EvalResults::new("p")
        .with_status("timeout")
        .raise_for_status(Some("custom"))
        .unwrap_err()
    else {
        unreachable!()
    };
    assert_eq!(msg, "custom");
}

#[test]
fn assert_score_at_least() {
    let scores = |s: f64| {
        vec![
            EvalScoreResult::new("rubric", s),
            EvalScoreResult::new("other", 0.1),
        ]
    };
    let sub = EvalResults::new("p").with_items(vec![item_result("s1", "pass", scores(0.5))]);
    let results = EvalResults::new("p")
        .with_items(vec![item_result("1", "pass", scores(0.9))])
        .with_sub_result("agent", sub);

    let Error::EvalNotPassed(msg) = results
        .assert_score_at_least(0.8, Some("rubric"), None)
        .unwrap_err()
    else {
        unreachable!()
    };
    assert_eq!(
        msg,
        "1 score(s) below threshold 0.8 for rubric: s1/rubric=0.500"
    );
    assert!(results
        .assert_score_at_least(0.5, Some("rubric"), None)
        .is_ok());

    let Error::EvalNotPassed(msg) = results.assert_score_at_least(1.0, None, None).unwrap_err()
    else {
        unreachable!()
    };
    assert!(msg.starts_with("4 score(s) below threshold 1.0: 1/rubric=0.900, 1/other=0.100"));

    let many = EvalResults::new("p").with_items(
        (0..7)
            .map(|i| item_result(&i.to_string(), "fail", vec![EvalScoreResult::new("r", 0.0)]))
            .collect(),
    );
    let Error::EvalNotPassed(msg) = many.assert_score_at_least(0.5, None, None).unwrap_err() else {
        unreachable!()
    };
    assert!(msg.ends_with("4/r=0.000 (+2 more)"), "{msg}");
    assert!(many
        .assert_score_at_least(0.5, None, Some("gate"))
        .unwrap_err()
        .to_string()
        .ends_with("gate"));
}

fn rubric(id: &str, score: Option<i64>, applicable: bool) -> RubricScore {
    RubricScore {
        id: id.into(),
        score,
        applicable,
        weight: 1,
        reason: String::new(),
    }
}

#[test]
fn assert_dimension_score_at_least() {
    let with_dims = |dims: Vec<RubricScore>| {
        let mut s = EvalScoreResult::new("rubric", 1.0);
        s.dimensions = Some(dims);
        vec![s]
    };
    let results = EvalResults::new("p").with_items(vec![
        item_result(
            "1",
            "pass",
            with_dims(vec![rubric("accuracy", Some(4), true)]),
        ),
        item_result(
            "2",
            "pass",
            with_dims(vec![rubric("accuracy", Some(2), true)]),
        ),
        item_result("3", "pass", with_dims(vec![rubric("accuracy", None, true)])),
        item_result(
            "4",
            "pass",
            with_dims(vec![rubric("accuracy", Some(0), false)]),
        ),
        item_result("5", "pass", vec![EvalScoreResult::new("plain", 1.0)]),
    ]);
    let Error::EvalNotPassed(msg) = results
        .assert_dimension_score_at_least("accuracy", 3.0, None, false, None)
        .unwrap_err()
    else {
        unreachable!()
    };
    assert_eq!(
        msg,
        "2 dimension score(s) for 'accuracy' below 3.0: 2/rubric/accuracy=2, 3/rubric/accuracy=None"
    );
    assert!(
        results
            .assert_dimension_score_at_least("accuracy", 0.0, Some("rubric"), false, None)
            .is_err(),
        "None scores always fail"
    );

    let only_good = EvalResults::new("p").with_items(vec![
        item_result(
            "1",
            "pass",
            with_dims(vec![rubric("accuracy", Some(4), true)]),
        ),
        item_result(
            "4",
            "pass",
            with_dims(vec![rubric("accuracy", Some(0), false)]),
        ),
    ]);
    assert!(only_good
        .assert_dimension_score_at_least("accuracy", 3.0, None, false, None)
        .is_ok());
    let Error::EvalNotPassed(msg) = only_good
        .assert_dimension_score_at_least("accuracy", 3.0, None, true, None)
        .unwrap_err()
    else {
        unreachable!()
    };
    assert_eq!(msg, "Dimension 'accuracy' not applicable on 1 item(s): 4");
    // An evaluator filter that matches nothing skips every score.
    assert!(only_good
        .assert_dimension_score_at_least("accuracy", 9.0, Some("nope"), false, None)
        .is_ok());
}

#[test]
fn assert_no_failed_items() {
    let results = EvalResults::new("p")
        .with_items(vec![
            item_result("1", "pass", vec![]),
            item_result("2", "fail", vec![]),
        ])
        .with_sub_result(
            "agent",
            EvalResults::new("p").with_items(vec![item_result("3", "errored", vec![])]),
        );
    let Error::EvalNotPassed(msg) = results.assert_no_failed_items(None).unwrap_err() else {
        unreachable!()
    };
    assert_eq!(msg, "2 item(s) failed or errored: 2:fail, 3:error");
    assert!(EvalResults::new("p")
        .with_items(vec![item_result("1", "pass", vec![])])
        .assert_no_failed_items(None)
        .is_ok());
}

#[test]
fn results_round_trip_through_json() {
    let results = EvalResults::new("p")
        .with_result_counts(ResultCounts::new(1, 0, 0))
        .with_items(vec![item_result(
            "1",
            "pass",
            vec![EvalScoreResult::new("r", 1.0)],
        )]);
    let v = serde_json::to_value(&results).unwrap();
    assert_eq!(v["status"], "completed");
    let back: EvalResults = serde_json::from_value(v).unwrap();
    assert_eq!(back, results);
}
