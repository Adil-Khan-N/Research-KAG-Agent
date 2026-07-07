"""
Hybrid KAG Research Assistant — Complete Streamlit Frontend
All 6 pages:
  1. 💬 Query          — standard + enhanced + checked + structured
  2. 📚 Literature Review — multi-agent pipeline
  3. 🕸️ Graph Explorer  — neighborhood + timeline + recommendations
  4. 📊 A/B Dashboard   — variant comparison from query_logs
  5. 📈 Analytics       — query logs, cache stats, slowest queries
  6. ⚙️ System Stats    — DB counts, papers list
"""

import streamlit as st
import requests
import os
import re
import time
import json
from collections import defaultdict

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Hybrid KAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 KAG Assistant")
    st.markdown("*Hybrid Knowledge-Augmented Generation*")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "💬 Query",
            "📚 Literature Review",
            "🕸️ Graph Explorer",
            "📊 A/B Dashboard",
            "📈 Analytics",
            "⚙️ System Stats",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # API health indicator
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        h = r.json()
        if h["status"] == "healthy":
            st.success("API ✓ Online")
        else:
            st.warning("API ⚠ Degraded")
        st.caption(
            f"📄 {h['paper_count']} papers | "
            f"🧩 {h['chunk_count']} chunks"
        )
    except Exception:
        st.error("API ✗ Offline")
        st.caption("Start: uvicorn api.main:app --port 8000")

    st.divider()
    st.caption("Vision Transformer Corpus")
    st.caption("25 papers · Neo4j + pgvector")
    st.caption("v2.0-full")


# ── Query-page helper functions ──────────────────────────────
# NOTE: these must be defined before the `if/elif` page-routing
# chain below, since the "Query" page calls them and Python
# resolves names at call time. Previously these were placed
# between the `if` and `elif` blocks, which is a SyntaxError
# (an if/elif chain must be contiguous) and would also have left
# the functions undefined when first called.
def _clean_answer(answer: str) -> str:
    answer = re.sub(r'\nCITATIONS:.*', '', answer, flags=re.DOTALL)
    answer = re.sub(r'\nCONFIDENCE:.*', '', answer, flags=re.DOTALL)
    return answer.strip()


def _conf_icon(conf: str) -> str:
    return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")


def _render_standard_result(r, elapsed):
    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    st.divider()
    conf = data.get("confidence", "low")
    st.markdown(
        f"### Answer {_conf_icon(conf)} *{conf} confidence* "
        f"— {data.get('total_latency_ms', 0)}ms"
    )
    st.markdown(_clean_answer(data.get("answer", "")))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{data.get('total_latency_ms', 0)}ms")
    m2.metric("Graph papers", data.get("graph_papers_found", 0))
    m3.metric("Graph boosted", data.get("graph_boosted_count", 0))
    m4.metric("Citations", len(data.get("citations", [])))

    variant = data.get("pipeline_variant", "")
    if variant == "cache_hit":
        st.info("⚡ Served from semantic cache")
    elif variant == "decomposed":
        st.info("🔀 Query was decomposed into sub-queries")

    citations = data.get("citations", [])
    if citations:
        st.markdown("### Sources")
        for c in citations:
            st.markdown(
                f"**[{c['citation_number']}]** {c['title']} "
                f"({c['year']}) — `arxiv:{c['arxiv_id']}`"
            )

    chunks = data.get("chunks_used", [])
    if chunks:
        with st.expander(f"📄 Evidence chunks ({len(chunks)})"):
            for chunk in chunks:
                badge = (
                    "🔵 Graph boosted"
                    if chunk.get("source") == "graph_boosted"
                    else "⚪ Vector only"
                )
                st.markdown(
                    f"**{chunk.get('title', '')[:50]}** "
                    f"({chunk.get('year', '')}) — "
                    f"*{chunk.get('section', '')}* — {badge}"
                )
                st.caption(
                    f"Rerank: {chunk.get('rerank_score', 0):.3f} | "
                    f"{chunk.get('retrieval_path', '')}"
                )
                st.markdown(f"> {chunk.get('text', '')[:250]}...")
                st.divider()


def _render_enhanced_result(r, elapsed):
    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    st.divider()
    conf = data.get("confidence", "low")
    st.markdown(
        f"### Answer {_conf_icon(conf)} *{conf} confidence* "
        f"— {data.get('total_latency_ms', 0)}ms"
    )
    st.markdown(_clean_answer(data.get("answer", "")))

    m1, m2, m3 = st.columns(3)
    m1.metric("Latency", f"{data.get('total_latency_ms', 0)}ms")
    m2.metric("Contradictions", data.get("contradiction_count", 0))
    m3.metric("Explanations", len(data.get("explanations", [])))

    # Contradictions
    contradictions = data.get("contradictions", [])
    if contradictions:
        st.warning(
            f"⚠️ {len(contradictions)} contradiction(s) detected in evidence"
        )
        for c in contradictions:
            with st.expander(
                f"⚔️ {c.get('paper_a', '')[:35]} "
                f"vs {c.get('paper_b', '')[:35]}"
            ):
                st.markdown(f"**Topic:** {c.get('topic', '')}")
                st.markdown(
                    f"**{c.get('paper_a', '')[:40]}** claims: "
                    f"{c.get('claim_a', '')}"
                )
                st.markdown(
                    f"**{c.get('paper_b', '')[:40]}** claims: "
                    f"{c.get('claim_b', '')}"
                )
                st.caption(
                    f"Confidence: {c.get('confidence', '')} | "
                    f"Method: {c.get('method', '')}"
                )

    # Retrieval explanations
    explanations = data.get("explanations", [])
    if explanations:
        with st.expander(
            f"🔍 Retrieval explanations ({len(explanations)} chunks)"
        ):
            for i, e in enumerate(explanations, 1):
                src = e.get("retrieval", {}).get("source", "")
                icon = "🔵" if src == "graph_boosted" else "⚪"
                scores = e.get("scores", {})
                retrieval = e.get("retrieval", {})
                rc = retrieval.get("rank_change", 0)
                arrow = (
                    f"↑{abs(rc)}" if rc < 0
                    else f"↓{rc}" if rc > 0 else "="
                )
                st.markdown(
                    f"**[{i}] {icon} {e.get('title', '')[:50]}** "
                    f"({e.get('year', '')}) — *{e.get('section', '')}*"
                )
                col_a, col_b, col_c = st.columns(3)
                col_a.caption(
                    f"vec={scores.get('vector', 0):.3f} | "
                    f"rerank={scores.get('rerank', 0):.3f}"
                )
                col_b.caption(
                    f"graph={scores.get('graph', 0):.1f} | "
                    f"final={scores.get('final', 0):.3f}"
                )
                col_c.caption(
                    f"rank #{retrieval.get('vector_rank', 0)} → "
                    f"#{retrieval.get('rerank_rank', 0)} ({arrow})"
                )
                st.caption(f"💡 {e.get('explanation', '')}")
                st.divider()


def _render_checked_result(r, elapsed):
    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    sc = data.get("self_check", {})
    st.divider()

    conf = data.get("confidence", "low")
    st.markdown(f"### Answer {_conf_icon(conf)} *{conf} confidence*")

    # Self-check badge
    if sc.get("is_faithful"):
        st.success(
            f"✅ Faithfulness check passed "
            f"(score: {sc.get('faithfulness_score', 0):.2f})"
        )
    else:
        st.error(
            f"⚠️ Faithfulness issues detected "
            f"(score: {sc.get('faithfulness_score', 0):.2f})"
        )

    # Show flagged answer if there are warnings
    if sc.get("has_warnings"):
        st.markdown("#### Answer (with unverified claims marked ⚠️)")
        flagged = data.get("flagged_answer", data.get("answer", ""))
        st.markdown(_clean_answer(flagged))
    else:
        st.markdown(_clean_answer(data.get("answer", "")))

    # Self-check details
    with st.expander("🔬 Self-check details"):
        st.markdown(f"**Verdict:** {sc.get('verdict', '')}")
        st.metric("Faithfulness score", f"{sc.get('faithfulness_score', 0):.2f}")
        st.metric("Check latency", f"{sc.get('latency_ms', 0)}ms")

        unsupported = sc.get("unsupported_claims", [])
        if unsupported:
            st.markdown("**Unsupported claims:**")
            for claim in unsupported:
                st.markdown(f"- ⚠️ {claim}")

        supported = sc.get("supported_claims", [])
        if supported:
            st.markdown("**Supported claims:**")
            for claim in supported[:5]:
                st.markdown(f"- ✅ {claim}")

    citations = data.get("citations", [])
    if citations:
        st.markdown("### Sources")
        for c in citations:
            st.markdown(
                f"**[{c.get('citation_number', '')}]** "
                f"{c.get('title', '')[:55]} ({c.get('year', '')}) — "
                f"`arxiv:{c.get('arxiv_id', '')}`"
            )


def _render_structured_result(r, schema_name, elapsed):
    if r.status_code != 200:
        st.error(f"Error {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    st.divider()
    st.markdown(
        f"### Structured Output — `{schema_name}` schema "
        f"({'✅ valid' if data.get('schema_valid') else '❌ invalid'})"
    )
    st.caption(
        f"Latency: {data.get('latency_ms', 0)}ms | "
        f"Graph boosted: {data.get('graph_boosted', 0)}"
    )

    if data.get("validation_errors"):
        for err in data["validation_errors"]:
            st.warning(f"Schema error: {err}")

    output = data.get("data")
    if output is None:
        st.error("No structured output returned.")
        return

    if schema_name == "comparison" and isinstance(output, dict):
        st.markdown(
            f"**Comparing:** {output.get('comparison_topic', '')}"
        )
        items = output.get("items", [])
        if items:
            cols = st.columns(len(items))
            for col, item in zip(cols, items):
                with col:
                    st.markdown(f"#### {item.get('name', 'Unknown')}")
                    st.caption(f"Year: {item.get('year', '—')}")
                    if item.get("key_feature"):
                        st.markdown(
                            f"**Key feature:** {item['key_feature']}"
                        )
                    if item.get("advantage"):
                        st.markdown(f"**Advantage:** {item['advantage']}")
                    if item.get("limitation"):
                        st.markdown(f"**Limitation:** {item['limitation']}")
                    if item.get("datasets"):
                        st.markdown(
                            f"**Datasets:** {', '.join(item['datasets'])}"
                        )
        if output.get("summary"):
            st.markdown(f"**Summary:** {output['summary']}")

    elif schema_name == "paper_list" and isinstance(output, list):
        for item in output:
            with st.container():
                st.markdown(
                    f"**[{item.get('year', '—')}] "
                    f"{item.get('title', 'Unknown')}**"
                )
                if item.get("contribution"):
                    st.caption(item["contribution"])
                cols = st.columns(3)
                if item.get("key_method"):
                    cols[0].caption(f"Method: {item['key_method']}")
                if item.get("dataset"):
                    cols[1].caption(f"Dataset: {item['dataset']}")
                if item.get("arxiv_id"):
                    cols[2].caption(f"arxiv: {item['arxiv_id']}")
                st.divider()

    else:
        st.json(output)


# ═══════════════════════════════════════════════════════════════
# PAGE 1: QUERY
# ═══════════════════════════════════════════════════════════════
if page == "💬 Query":
    st.title("💬 Research Query")
    st.markdown(
        "Ask questions about the Vision Transformer corpus. "
        "Choose a query mode below."
    )

    # Mode selector
    mode = st.radio(
        "Query mode",
        ["Standard", "Enhanced (+ contradictions + explanations)",
         "Checked (+ hallucination self-check)",
         "Structured (+ JSON schema output)"],
        horizontal=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="How does shifted window attention work in Swin Transformer?",
            label_visibility="collapsed",
        )
    with col2:
        use_graph = st.checkbox("Use graph", value=True)

    # Example queries
    st.markdown("**Try these:**")
    ex_cols = st.columns(4)
    examples = [
        "What is the key innovation of ViT?",
        "Compare ViT and Swin Transformer",
        "What datasets do papers extending ViT use?",
        "How does DeiT improve data efficiency?",
    ]
    for i, (col, ex) in enumerate(zip(ex_cols, examples)):
        if col.button(ex, key=f"ex_{i}"):
            query = ex

    # Structured schema selector
    schema_name = ""
    if mode == "Structured (+ JSON schema output)":
        schema_name = st.selectbox(
            "Output schema",
            ["comparison", "paper_list", "metric", "timeline"],
        )

    if query and st.button("🔍 Search", type="primary"):
        with st.spinner("Retrieving and generating..."):
            t0 = time.time()
            try:
                # Choose endpoint based on mode
                if mode == "Standard":
                    r = requests.post(
                        f"{API_URL}/query",
                        json={"query": query, "use_graph": use_graph},
                        timeout=120,
                    )
                    _render_standard_result(r, time.time() - t0)

                elif mode == "Enhanced (+ contradictions + explanations)":
                    r = requests.post(
                        f"{API_URL}/query-enhanced",
                        json={"query": query, "use_graph": use_graph},
                        timeout=180,
                    )
                    _render_enhanced_result(r, time.time() - t0)

                elif mode == "Checked (+ hallucination self-check)":
                    r = requests.post(
                        f"{API_URL}/query-checked",
                        json={"query": query, "use_graph": use_graph},
                        timeout=180,
                    )
                    _render_checked_result(r, time.time() - t0)

                elif mode == "Structured (+ JSON schema output)":
                    r = requests.post(
                        f"{API_URL}/query-structured",
                        json={"query": query, "schema": schema_name},
                        timeout=120,
                    )
                    _render_structured_result(r, schema_name, time.time() - t0)

            except requests.exceptions.Timeout:
                st.error("Request timed out. Models may be loading — try again.")
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: LITERATURE REVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "📚 Literature Review":
    st.title("📚 Literature Review Generator")
    st.markdown(
        "Generate a structured literature review using the "
        "5-agent pipeline: Planner → PaperSearch → "
        "Summary → Graph → Writer"
    )

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. vision transformers for image classification",
    )

    st.markdown("**Example topics:**")
    t_cols = st.columns(3)
    topics = [
        "masked autoencoder pretraining",
        "hierarchical vision transformers",
        "efficient vision transformers for mobile",
    ]
    for i, (col, t) in enumerate(zip(t_cols, topics)):
        if col.button(t, key=f"topic_{i}"):
            topic = t

    if topic and st.button("🚀 Generate Literature Review", type="primary"):
        progress = st.progress(0)
        status   = st.empty()

        try:
            import sys
            sys.path.insert(0, ".")

            from agents.planner      import planner_agent
            from agents.paper_search import paper_search_agent
            from agents.summary      import summary_agent
            from agents.graph_agent  import graph_agent
            from agents.writer       import writer_agent
            from agents.pipeline     import save_literature_review

            state = {
                "topic": topic,
                "sub_queries": [], "plan_reasoning": "",
                "retrieved_papers": [], "retrieved_chunks": [],
                "paper_summaries": [], "graph_relationships": [],
                "timeline": [], "literature_review": "",
                "citations": [], "total_papers_found": 0, "errors": [],
            }

            status.info("🧠 Step 1/5: PlannerAgent — decomposing topic...")
            progress.progress(10)
            state = planner_agent(state)
            st.caption(
                f"✓ Generated {len(state['sub_queries'])} queries: "
                f"{state['sub_queries'][:2]}"
            )

            status.info("🔍 Step 2/5: PaperSearchAgent — retrieving papers...")
            progress.progress(30)
            state = paper_search_agent(state)
            st.caption(f"✓ Found {state['total_papers_found']} papers")

            status.info("📝 Step 3/5: SummaryAgent — summarizing...")
            progress.progress(50)
            state = summary_agent(state)
            st.caption(
                f"✓ Summarized {len(state['paper_summaries'])} papers"
            )

            status.info("🕸️ Step 4/5: GraphAgent — finding relationships...")
            progress.progress(70)
            state = graph_agent(state)
            st.caption(
                f"✓ {len(state['graph_relationships'])} relationships, "
                f"{len(state['timeline'])} timeline entries"
            )

            status.info("✍️ Step 5/5: WriterAgent — synthesizing review...")
            progress.progress(85)
            state = writer_agent(state)

            filename = save_literature_review(state)
            progress.progress(100)
            status.success("✅ Literature review complete!")

            # Results
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Papers reviewed", state["total_papers_found"])
            c2.metric("Relationships", len(state["graph_relationships"]))
            c3.metric("Citations", len(state["citations"]))

            st.markdown("### Generated Review")
            st.markdown(state["literature_review"])

            if state.get("timeline"):
                with st.expander("📅 Paper Timeline"):
                    for t in state["timeline"]:
                        m_str = (
                            " — " + ", ".join(t["methods"][:2])
                            if t.get("methods") else ""
                        )
                        st.markdown(f"**{t['year']}** — {t['title']}{m_str}")

            with st.expander("🔍 Search queries used"):
                for q in state.get("sub_queries", []):
                    st.markdown(f"- {q}")

            if state.get("errors"):
                with st.expander("⚠️ Warnings"):
                    for e in state["errors"]:
                        st.warning(str(e)[:150])

            st.download_button(
                "⬇️ Download Markdown",
                data=state["literature_review"],
                file_name=f"lit_review_{topic[:30].replace(' ','_')}.md",
                mime="text/markdown",
            )

        except Exception as e:
            progress.progress(0)
            status.error(f"Pipeline error: {e}")
            import traceback
            with st.expander("Full error"):
                st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════
# PAGE 3: GRAPH EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "🕸️ Graph Explorer":
    st.title("🕸️ Knowledge Graph Explorer")

    tab1, tab2, tab3 = st.tabs(
        ["📍 Paper Neighborhood", "⏳ Timeline", "⭐ Recommendations"]
    )

    # ── Tab 1: Paper neighborhood ─────────────────────────────
    with tab1:
        st.markdown("Explore the graph neighborhood of a paper or entity.")

        col1, col2 = st.columns([2, 1])
        with col1:
            entity = st.text_input(
                "arXiv ID or entity name",
                value="2010.11929",
                help="e.g. 2010.11929 for ViT, or 'ImageNet' for a dataset",
            )
        with col2:
            entity_type = st.selectbox(
                "Type", ["paper", "dataset", "method", "concept"]
            )

        st.markdown("**Quick access:**")
        q_cols = st.columns(5)
        quick = [
            ("ViT",  "2010.11929"),
            ("Swin", "2103.14030"),
            ("DeiT", "2012.12877"),
            ("MAE",  "2111.06377"),
            ("DETR", "2005.12872"),
        ]
        for col, (name, aid) in zip(q_cols, quick):
            if col.button(name, key=f"q_{name}"):
                entity = aid
                entity_type = "paper"

        if entity:
            with st.spinner("Fetching graph neighborhood..."):
                try:
                    r = requests.get(
                        f"{API_URL}/graph-explore",
                        params={
                            "entity": entity,
                            "entity_type": entity_type,
                        },
                        timeout=15,
                    )
                    if r.status_code == 200:
                        gdata = r.json()
                        summary = gdata["neighborhood_summary"]
                        nodes = gdata["nodes"]
                        edges = gdata["edges"]

                        st.divider()
                        if entity_type == "paper":
                            m_cols = st.columns(6)
                            metrics = [
                                ("Methods",     summary.get("methods_count", 0)),
                                ("Datasets",    summary.get("datasets_count", 0)),
                                ("Concepts",    summary.get("concepts_count", 0)),
                                ("Tasks",       summary.get("tasks_count", 0)),
                                ("Extends",     summary.get("extends_count", 0)),
                                ("Extended by", summary.get("extended_by_count", 0)),
                            ]
                            for col, (label, val) in zip(m_cols, metrics):
                                col.metric(label, val)
                        else:
                            st.metric(
                                "Connected papers",
                                summary.get("connected_papers", 0),
                            )

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Nodes ({len(nodes)})**")
                            icons = {
                                "paper": "📄", "method": "⚙️",
                                "dataset": "📊", "concept": "💡",
                                "task": "🎯",
                            }
                            for node in nodes[:20]:
                                icon = icons.get(node["type"], "•")
                                st.markdown(
                                    f"{icon} **{node['label'][:40]}** "
                                    f"*({node['type']})*"
                                )

                        with col_b:
                            st.markdown(f"**Relationships ({len(edges)})**")
                            for edge in edges[:20]:
                                st.markdown(
                                    f"→ `{edge['relationship']}`: "
                                    f"{edge['target'][:35]}"
                                )

                        # Plotly visualization
                        if len(nodes) > 1:
                            try:
                                import plotly.graph_objects as go
                                import math

                                n = len(nodes)
                                pos = {}
                                for i, node in enumerate(nodes):
                                    angle = 2 * math.pi * i / n
                                    radius = 0 if i == 0 else 1
                                    pos[node["id"]] = (
                                        radius * math.cos(angle),
                                        radius * math.sin(angle),
                                    )

                                ex_list, ey_list = [], []
                                for edge in edges:
                                    if (edge["source"] in pos
                                            and edge["target"] in pos):
                                        x0, y0 = pos[edge["source"]]
                                        x1, y1 = pos[edge["target"]]
                                        ex_list += [x0, x1, None]
                                        ey_list += [y0, y1, None]

                                colors = {
                                    "paper": "#4f8ef7",
                                    "method": "#f7a74f",
                                    "dataset": "#4ff7a7",
                                    "concept": "#f74f8e",
                                    "task": "#a74ff7",
                                }
                                nx_list = [pos[nd["id"]][0] for nd in nodes if nd["id"] in pos]
                                ny_list = [pos[nd["id"]][1] for nd in nodes if nd["id"] in pos]
                                nt_list = [nd["label"][:20] for nd in nodes if nd["id"] in pos]
                                nc_list = [colors.get(nd["type"], "#ccc") for nd in nodes if nd["id"] in pos]

                                fig = go.Figure(
                                    data=[
                                        go.Scatter(
                                            x=ex_list, y=ey_list,
                                            mode="lines",
                                            line=dict(width=1, color="#aaa"),
                                            hoverinfo="none",
                                        ),
                                        go.Scatter(
                                            x=nx_list, y=ny_list,
                                            mode="markers+text",
                                            text=nt_list,
                                            textposition="top center",
                                            marker=dict(
                                                size=14,
                                                color=nc_list,
                                                line=dict(width=1, color="white"),
                                            ),
                                            hoverinfo="text",
                                        ),
                                    ],
                                    layout=go.Layout(
                                        showlegend=False,
                                        hovermode="closest",
                                        margin=dict(b=20, l=5, r=5, t=20),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        height=400,
                                    ),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.caption(f"Visualization unavailable: {e}")
                    elif r.status_code == 404:
                        st.warning(f"Entity not found: {entity}")
                    else:
                        st.error(f"Error {r.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: Timeline ───────────────────────────────────────
    with tab2:
        st.markdown("Trace the chronological evolution of a paper lineage.")

        col1, col2 = st.columns(2)
        with col1:
            tl_arxiv = st.text_input(
                "arXiv ID", value="2010.11929",
                help="Seed paper for EXTENDS chain traversal"
            )
        with col2:
            tl_concept = st.text_input(
                "Or concept", placeholder="self-supervised learning"
            )

        tl_depth = st.slider("Max depth (hops)", 1, 5, 3)

        if st.button("⏳ Build Timeline"):
            with st.spinner("Traversing EXTENDS chain..."):
                try:
                    params = {"max_depth": tl_depth}
                    if tl_arxiv:
                        params["arxiv_id"] = tl_arxiv
                    if tl_concept:
                        params["concept"] = tl_concept

                    r = requests.get(
                        f"{API_URL}/timeline",
                        params=params,
                        timeout=20,
                    )
                    if r.status_code == 200:
                        tdata = r.json()
                        entries = tdata.get("entries", [])
                        yr = tdata.get("year_range", {})

                        st.success(
                            f"{tdata['total_papers']} papers "
                            f"({yr.get('start', '?')}–{yr.get('end', '?')}) "
                            f"in {tdata['latency_ms']}ms"
                        )

                        for entry in entries:
                            with st.container():
                                col_y, col_t = st.columns([1, 5])
                                col_y.markdown(f"**{entry['year']}**")
                                col_t.markdown(
                                    f"**{entry['title'][:60]}**"
                                )
                                if entry.get("contribution"):
                                    st.caption(
                                        entry["contribution"][:150]
                                    )
                                if entry.get("methods"):
                                    st.caption(
                                        "Methods: "
                                        + ", ".join(entry["methods"][:3])
                                    )
                                if entry.get("extends"):
                                    st.caption(
                                        "Extends: "
                                        + ", ".join(
                                            e[:30] for e in entry["extends"]
                                        )
                                    )
                                st.divider()
                    else:
                        st.error(f"Error {r.status_code}: {r.text[:100]}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 3: Recommendations ────────────────────────────────
    with tab3:
        st.markdown(
            "Get paper recommendations using pure graph traversal — "
            "**zero LLM calls**, sub-100ms."
        )

        rec_arxiv = st.text_input(
            "Seed paper arXiv ID", value="2010.11929"
        )
        rec_k = st.slider("Top K", 3, 10, 5)

        if st.button("⭐ Get Recommendations"):
            with st.spinner("Running graph traversal..."):
                try:
                    r = requests.get(
                        f"{API_URL}/recommend",
                        params={"arxiv_id": rec_arxiv, "top_k": rec_k},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        rdata = r.json()
                        st.success(
                            f"{len(rdata['recommendations'])} recommendations "
                            f"in {rdata['latency_ms']}ms | "
                            f"LLM calls: {rdata['llm_calls']}"
                        )

                        for rec in rdata["recommendations"]:
                            with st.expander(
                                f"⭐ {rec['score']:.1f} — "
                                f"{rec['title'][:55]} ({rec['year']})"
                            ):
                                st.markdown(
                                    f"**Why:** {rec['reasoning']}"
                                )
                                cols = st.columns(3)
                                if rec.get("shared_methods"):
                                    cols[0].markdown(
                                        "**Methods:** "
                                        + ", ".join(
                                            rec["shared_methods"][:3]
                                        )
                                    )
                                if rec.get("shared_datasets"):
                                    cols[1].markdown(
                                        "**Datasets:** "
                                        + ", ".join(
                                            rec["shared_datasets"][:3]
                                        )
                                    )
                                badges = []
                                if rec.get("extends"):
                                    badges.append("↗ Extends seed")
                                if rec.get("extended_by"):
                                    badges.append("↙ Seed extends this")
                                if badges:
                                    cols[2].markdown(" | ".join(badges))
                                st.code(f"arxiv:{rec['arxiv_id']}")
                    else:
                        st.error(f"Error {r.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.markdown("#### Entity-based search")
        ent_name = st.text_input("Entity name", placeholder="ImageNet")
        ent_type = st.selectbox(
            "Entity type", ["dataset", "method", "concept", "task"]
        )
        if st.button("🔍 Find related papers"):
            try:
                r = requests.get(
                    f"{API_URL}/recommend/entity",
                    params={"entity": ent_name, "entity_type": ent_type},
                    timeout=10,
                )
                if r.status_code == 200:
                    papers = r.json().get("papers", [])
                    st.markdown(f"**{len(papers)} papers** related to `{ent_name}`:")
                    for p in papers:
                        st.markdown(
                            f"- [{p.get('year', '?')}] "
                            f"{p.get('title', '')[:60]}"
                        )
            except Exception as e:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════
# PAGE 4: A/B DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📊 A/B Dashboard":
    st.title("📊 A/B Pipeline Comparison Dashboard")
    st.markdown(
        "Compares **vector_only** vs **hybrid** pipeline performance "
        "from logged query results."
    )

    refresh = st.button("🔄 Refresh data")
    if refresh:
        st.rerun()

    try:
        r = requests.get(f"{API_URL}/ab-summary", timeout=10)
        if r.status_code != 200:
            st.error(f"Could not load A/B data: {r.status_code}")
            st.stop()

        data = r.json()
        stats = {
            s["pipeline_variant"]: s
            for s in data.get("stats_by_variant", [])
        }
        paired = data.get("paired_comparisons", [])
        recent = data.get("recent_queries", [])

    except Exception as e:
        st.error(f"API not reachable: {e}")
        st.stop()

    # ── Top metrics ───────────────────────────────────────────
    st.markdown("### Performance Summary")
    variants = ["hybrid", "vector_only", "cache_hit"]
    labels   = {"hybrid": "Hybrid KAG", "vector_only": "Vector Only",
                 "cache_hit": "Cache Hit"}
    cols = st.columns(len([v for v in variants if v in stats]) + 1)

    for col, variant in zip(cols, [v for v in variants if v in stats]):
        s = stats[variant]
        col.metric(
            labels.get(variant, variant),
            f"{s['query_count']} queries",
            f"avg {s['avg_latency_ms']}ms",
        )

    # Cache hit rate
    total = sum(s["query_count"] for s in stats.values())
    cache_hits = stats.get("cache_hit", {}).get("query_count", 0)
    hit_rate = cache_hits / total if total > 0 else 0
    cols[-1].metric("Cache hit rate", f"{hit_rate:.1%}", f"{cache_hits} hits")

    st.divider()

    # ── Latency comparison ─────────────────────────────────────
    st.markdown("### Latency Comparison")
    lat_cols = st.columns(2)

    with lat_cols[0]:
        st.markdown("**Average latency by variant**")
        for v in ["hybrid", "vector_only", "cache_hit"]:
            if v in stats:
                s = stats[v]
                max_ms = max(
                    stats[vv]["avg_latency_ms"]
                    for vv in stats if stats[vv]["avg_latency_ms"]
                ) or 1
                pct = s["avg_latency_ms"] / max_ms
                st.markdown(f"**{labels.get(v, v)}**")
                st.progress(
                    min(pct, 1.0),
                    text=(
                        f"avg={s['avg_latency_ms']}ms | "
                        f"median={s['median_latency']}ms | "
                        f"max={s['max_latency']}ms"
                    ),
                )

    with lat_cols[1]:
        st.markdown("**Chunk retrieval**")
        for v in ["hybrid", "vector_only"]:
            if v in stats:
                s = stats[v]
                st.metric(
                    labels.get(v, v),
                    f"{s['avg_chunks']:.1f} avg chunks",
                )

    st.divider()

    # ── Paired comparison table ───────────────────────────────
    st.markdown(f"### Paired Query Comparison ({len(paired)} queries)")

    if paired:
        hybrid_faster  = sum(1 for p in paired if p.get("hybrid_faster", False))
        vector_faster  = len(paired) - hybrid_faster

        w_cols = st.columns(3)
        w_cols[0].metric("Hybrid faster", hybrid_faster)
        w_cols[1].metric("Vector faster", vector_faster)
        w_cols[2].metric(
            "Avg latency delta",
            f"{sum(p.get('hybrid_ms', 0) - p.get('vector_ms', 0) for p in paired) // len(paired)}ms",
        )

        for p in paired[:8]:
            with st.expander(f"🔍 {p.get('query', '')[:65]}"):
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown("**🔵 Hybrid KAG**")
                    st.caption(f"Latency: {p.get('hybrid_ms', 0)}ms")
                    st.markdown(
                        p.get("hybrid_answer", "—")[:200] + "..."
                    )
                with pc2:
                    st.markdown("**⚪ Vector Only**")
                    st.caption(f"Latency: {p.get('vector_ms', 0)}ms")
                    st.markdown(
                        p.get("vector_answer", "—")[:200] + "..."
                    )
    else:
        st.info(
            "No paired comparisons yet. Run A/B tests using "
            "POST /ab-test or python scratch/run_ragas_eval.py"
        )
        if st.button("▶️ Run quick A/B test"):
            with st.spinner("Running A/B comparison (takes 2-3 min)..."):
                try:
                    r = requests.post(
                        f"{API_URL}/ab-test",
                        json={"query": "how does ViT handle image patches?"},
                        timeout=300,
                    )
                    if r.status_code == 200:
                        result = r.json()
                        st.success(
                            f"Winner: **{result['winner'].upper()}** — "
                            f"{result['reason']}"
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"A/B test failed: {e}")

    st.divider()

    # ── Recent queries ────────────────────────────────────────
    st.markdown("### Recent Queries")
    if recent:
        for q in recent[:10]:
            badge = {
                "hybrid": "🔵",
                "vector_only": "⚪",
                "cache_hit": "⚡",
                "decomposed": "🔀",
            }.get(q.get("variant", ""), "•")

            with st.expander(
                f"{badge} {q.get('query', '')[:60]} "
                f"— {q.get('latency_ms', 0)}ms"
            ):
                st.markdown(f"**Variant:** `{q.get('variant', '')}`")
                st.markdown(
                    f"**Answer:** {q.get('answer', '')[:200]}..."
                )
                st.caption(f"Time: {q.get('created_at', '')}")
    else:
        st.info("No queries logged yet. Use the Query page to get started.")


# ═══════════════════════════════════════════════════════════════
# PAGE 5: ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.title("📈 Query Analytics")

    try:
        r = requests.get(f"{API_URL}/analytics?limit=10", timeout=5)
        if r.status_code != 200:
            st.error("Analytics endpoint unavailable")
            st.stop()
        data = r.json()
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    # ── Top metrics ───────────────────────────────────────────
    a_cols = st.columns(4)
    a_cols[0].metric("Total queries", data.get("total_queries", 0))
    a_cols[1].metric(
        "Cache hit rate", f"{data.get('cache_hit_rate', 0):.1%}"
    )
    variants = data.get("variant_distribution", [])
    hybrid_count = next(
        (v["count"] for v in variants if v["variant"] == "hybrid"), 0
    )
    a_cols[2].metric("Hybrid queries", hybrid_count)
    a_cols[3].metric("Variants tracked", len(variants))

    st.divider()

    # ── Variant distribution ──────────────────────────────────
    st.markdown("### Pipeline Variant Distribution")
    total = data.get("total_queries", 1) or 1
    for v in variants:
        pct = v["count"] / total
        label = {
            "hybrid":       "🔵 Hybrid KAG",
            "vector_only":  "⚪ Vector Only",
            "cache_hit":    "⚡ Cache Hit",
            "decomposed":   "🔀 Decomposed",
        }.get(v["variant"], v["variant"])
        st.progress(
            pct,
            text=f"{label}: {v['count']} queries ({pct:.1%})",
        )

    st.divider()

    # ── Top queries ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Most Frequent Queries")
        top = data.get("top_queries", [])
        if top:
            for q in top[:8]:
                st.markdown(
                    f"**{q['count']}x** — `{q['query'][:55]}`"
                )
                st.caption(f"avg {q['avg_ms']}ms")
        else:
            st.info("No queries yet.")

    with col2:
        st.markdown("### 🐢 Slowest Queries")
        slow = data.get("slowest_queries", [])
        if slow:
            for q in slow[:8]:
                st.markdown(f"**{q['ms']}ms** — `{q['query'][:50]}`")
                st.caption(f"variant: {q['variant']}")
        else:
            st.info("No slow queries logged.")

    st.divider()

    # ── Latency trend ─────────────────────────────────────────
    st.markdown("### ⏱️ Latency Trend (last 20 queries)")
    trend = data.get("latency_trend", [])
    if trend:
        try:
            import plotly.graph_objects as go

            hybrid_pts = [
                t for t in trend if t["variant"] == "hybrid"
            ]
            vector_pts = [
                t for t in trend if t["variant"] == "vector_only"
            ]

            fig = go.Figure()
            if hybrid_pts:
                fig.add_trace(go.Scatter(
                    y=[p["ms"] for p in hybrid_pts],
                    mode="lines+markers",
                    name="Hybrid KAG",
                    line=dict(color="#4f8ef7"),
                ))
            if vector_pts:
                fig.add_trace(go.Scatter(
                    y=[p["ms"] for p in vector_pts],
                    mode="lines+markers",
                    name="Vector Only",
                    line=dict(color="#aaa"),
                ))

            fig.update_layout(
                yaxis_title="Latency (ms)",
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            for t in trend[:10]:
                st.markdown(
                    f"- {t['variant']}: {t['ms']}ms"
                )

    # ── Cache stats ───────────────────────────────────────────
    st.divider()
    st.markdown("### ⚡ Cache Performance")
    try:
        rc = requests.get(f"{API_URL}/cache-stats", timeout=5)
        if rc.status_code == 200:
            cdata = rc.json()
            cc1, cc2 = st.columns(2)
            cc1.metric("Cached queries", cdata.get("total_cached", 0))
            cc2.metric("Total hits", cdata.get("total_hits", 0))

            top_cached = cdata.get("top_queries", [])
            if top_cached:
                st.markdown("**Most-hit cached queries:**")
                for cq in top_cached[:5]:
                    st.markdown(
                        f"- **{cq['hits']} hits** — `{cq['query'][:55]}`"
                    )
    except Exception:
        st.caption("Cache stats unavailable")

    if st.button("🗑️ Clear cache"):
        try:
            rd = requests.delete(f"{API_URL}/cache", timeout=5)
            if rd.status_code == 200:
                st.success("Cache cleared")
                st.rerun()
        except Exception as e:
            st.error(str(e))


# ═══════════════════════════════════════════════════════════════
# PAGE 6: SYSTEM STATS
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ System Stats":
    st.title("⚙️ System Statistics")

    col1, col2 = st.columns(2)

    # DB stats
    with col1:
        st.markdown("### 🗄️ Database Stats")
        try:
            r = requests.get(f"{API_URL}/stats", timeout=5)
            if r.status_code == 200:
                stats = r.json()
                pg = stats["postgres"]
                st.markdown("**PostgreSQL**")
                st.metric("Papers",        pg.get("papers", 0))
                st.metric("Chunks",        pg.get("chunks", 0))
                st.metric("Queries logged",pg.get("queries_logged", 0))

                st.markdown("**Neo4j — Nodes**")
                for label, count in stats["neo4j"]["nodes"].items():
                    st.metric(label, count)

                st.markdown("**Neo4j — Relationships**")
                for rel, count in stats["neo4j"]["relationships"].items():
                    st.metric(rel, count)
        except Exception as e:
            st.error(f"Stats unavailable: {e}")

    # Recent query logs
    with col2:
        st.markdown("### 📋 Recent Query Logs")
        try:
            r = requests.get(
                f"{API_URL}/query-logs",
                params={"limit": 10},
                timeout=5,
            )
            if r.status_code == 200:
                logs = r.json()
                st.caption(f"Total: {logs['total']} queries")
                for log in logs["logs"]:
                    badge = {
                        "hybrid": "🔵", "vector_only": "⚪",
                        "cache_hit": "⚡", "decomposed": "🔀",
                    }.get(log.get("pipeline_variant", ""), "•")
                    with st.expander(
                        f"{badge} {log['query'][:50]} "
                        f"— {log['latency_ms']}ms"
                    ):
                        st.markdown(f"**Answer:** {log['answer_preview']}...")
                        st.caption(
                            f"Variant: {log['pipeline_variant']} | "
                            f"Time: {log['created_at']}"
                        )
        except Exception as e:
            st.error(f"Logs unavailable: {e}")

    # Async ingestion jobs
    st.divider()
    st.markdown("### 📥 Ingestion Jobs")
    try:
        r = requests.get(f"{API_URL}/ingest/jobs", timeout=5)
        if r.status_code == 200:
            jobs = r.json().get("jobs", [])
            if jobs:
                for j in jobs[:5]:
                    status_icon = {
                        "complete": "✅", "processing": "⏳",
                        "failed": "❌", "queued": "🕐",
                    }.get(j.get("status", ""), "•")
                    st.markdown(
                        f"{status_icon} **{j['filename']}** — "
                        f"{j['status']} ({j['progress']}%)"
                    )
                    if j.get("result"):
                        st.caption(str(j["result"])[:100])
                    if j.get("error"):
                        st.caption(f"Error: {j['error'][:80]}")
            else:
                st.info("No ingestion jobs yet.")
    except Exception as e:
        st.caption(f"Jobs unavailable: {e}")

    # Papers list
    st.divider()
    st.markdown("### 📄 Ingested Papers")
    try:
        r = requests.get(f"{API_URL}/papers", timeout=5)
        if r.status_code == 200:
            pdata = r.json()
            st.caption(f"{pdata['total']} papers in corpus")

            by_year = defaultdict(list)
            for p in pdata["papers"]:
                by_year[p.get("year", 0)].append(p)

            for year in sorted(by_year.keys(), reverse=True):
                st.markdown(f"**{year}**")
                for p in by_year[year]:
                    st.markdown(
                        f"- {p['title'][:65]} "
                        f"(`{p['arxiv_id']}`, "
                        f"{p.get('total_chunks', 0)} chunks)"
                    )
    except Exception as e:
        st.error(f"Papers unavailable: {e}")