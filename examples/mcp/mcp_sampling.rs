//! MCP sampling: let an MCP *server* call back into *your* model.
//! `chat_client_sampling_handler` adapts any `ChatClient` into a
//! `SamplingHandler`; attaching it to an MCP tool advertises the `sampling`
//! capability during `initialize`, and the client then answers the server's
//! `sampling/createMessage` requests with real completions.
//!
//! Prerequisites: OPENAI_API_KEY (skips gracefully when unset) and a working
//! `npx` -- this spawns `@modelcontextprotocol/server-everything`, whose
//! sampling demo tool exercises exactly this callback. The tool's name has
//! changed across server releases (`sampleLLM` in older versions,
//! `simulate-research-query` in newer ones), so the prompt below asks the
//! model to pick whichever sampling-demo tool the server advertises.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example mcp_sampling
//! ```

use std::sync::Arc;

use agent_framework::mcp::{chat_client_sampling_handler_with, SamplingGuard};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(client) = OpenAIChatCompletionClient::from_env("gpt-4o-mini") else {
        println!("set OPENAI_API_KEY to run this example");
        return Ok(());
    };

    // The handler holds its own client; the server's sampling requests do
    // NOT go through the agent below (that's the point -- the server drives
    // these calls, with the host app deciding which model answers them).
    //
    // Sampling is denied unless a guard approves it: an MCP server is an
    // untrusted third party. This example opts in to everything, with the
    // default 4096-token cap and 25-request limit still applied.
    let handler = chat_client_sampling_handler_with(
        Arc::new(client.clone()) as Arc<dyn ChatClient>,
        SamplingGuard::default().approve_all(),
    );

    let mcp = McpStdioTool::new("everything", "npx")
        .args(["-y", "@modelcontextprotocol/server-everything"])
        .sampling_handler(handler); // <- advertises the sampling capability

    let tools = mcp.tool_definitions().await?;
    println!("connected; {} tool(s) discovered", tools.len());

    let agent = Agent::builder(client)
        .instructions("Use the available tools when asked.")
        .tools(tools)
        .build();

    // Invoking the server's sampling-demo tool makes it issue a
    // sampling/createMessage back to us mid-tool-call; our handler answers
    // it with a real OpenAI completion, and the tool result flows back.
    let response = agent
        .run_once(
            "Call the tool that demonstrates LLM sampling (it is named \
             `sampleLLM` or `simulate-research-query` depending on the server \
             version) and ask it: what color is the sky?",
        )
        .await?;
    println!("{}", response.text());

    mcp.close().await?;
    Ok(())
}
