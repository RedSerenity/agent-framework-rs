//! Embedded DevUI debug page.
//!
//! A single, self-contained HTML+CSS+vanilla-JS page (no external assets, no
//! CDNs, no build step) served by [`AgentHost`](crate::AgentHost) at `GET /`
//! and `GET /ui`. It fetches the entity list from `GET /v1/entities`, lets you
//! pick an entity, and runs it through `POST /v1/responses` — rendering the SSE
//! stream incrementally (live `output_text` deltas, collapsible executor /
//! workflow-event rows, and an inline notice for pending `request_info` events).
//! The page itself does not answer those requests; a client resumes the run by
//! posting a `workflow_hil_response` on the run's conversation (see
//! [`crate::devui`]). A non-stream toggle
//! posts with `stream:false` and renders the aggregated JSON response instead.
//!
//! This is a **pragmatic debug UI**, deliberately minimal — it is *not* the
//! upstream React DevUI, which ships as a separate bundled frontend. The page
//! embeds no dependencies and talks only to the same-origin DevUI API.

use axum::response::Html;
use axum::routing::get;
use axum::Router;

/// The embedded page, compiled into the binary. Self-contained: no external
/// URLs, scripts, styles, or fonts.
pub(crate) const PAGE: &str = include_str!("devui/page.html");

/// The `GET /` + `GET /ui` handler.
pub(crate) async fn ui_page() -> Html<&'static str> {
    Html(PAGE)
}

/// A stateless router serving the debug page at `/` and `/ui`.
pub(crate) fn router() -> Router {
    Router::new()
        .route("/", get(ui_page))
        .route("/ui", get(ui_page))
}
