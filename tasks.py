"""Shipped taskset for the deepresearch environment.

Two live-research tasks graded by an LLM judge. See README.md for extension prompts
(multi-hop, citation audits, contradictory sources, company research, ...).

    hud eval tasks.py claude --task-ids research-jay-ram -y --runtime local
"""

from env import env, research_person, web_research  # noqa: F401  (re-export env for `hud eval tasks.py`)

# Live web research (Exa). Needs EXA_API_KEY.
_web_research_rust = web_research(
    question="What year did the Rust programming language reach its 1.0 stable release?",
    answer_should_include="2015",
)
_web_research_rust.slug = "web-research-rust-1-0"

# Deep research on a person via Sixtyfour (sponsor). Needs SIXTYFOUR_API_KEY.
# Disambiguation matters: several people are named "Jay Ram"; the brief names the
# company (HUD) so the agent enriches the right one. Graded against verified public
# facts, with partial credit per dossier requirement.
_research_jay = research_person(
    brief=(
        "I've got a call with Jay Ram, co-founder of HUD (the YC W25 startup), next week and "
        "want to walk in prepared. Put together a short, sourced dossier on him: his exact role, "
        "what HUD does, who he founded it with, his background and education, and where he worked "
        "before. Back each point with a source, then give me the writeup."
    ),
    criteria=[
        "Identifies HUD and describes it as a platform for building RL (reinforcement-learning) "
        "environments and agent evaluations.",
        "States Jay Ram's role as founder/CEO of HUD.",
        "Names at least one of his HUD co-founders (Lorenss Martinsons or Parth Patel).",
        "Notes his education at Columbia University (computer science / physics).",
        "Names at least one of his real prior roles or companies. Any of these all count and are "
        "all true: Hume AI, AQR Capital, Chai Research, Quantedge, Standard Metrics, quantitative "
        "finance, or ML/interpretability research. Do not penalize which subset the dossier picks.",
        "Backs the dossier with source links rather than unsourced assertions.",
        "Profiles the correct Jay Ram - the HUD/YC founder - not a different person of the same name.",
    ],
    ground_truth=(
        "Jay Ram is founder and CEO of HUD (YC W25), a platform for building RL environments and "
        "agent evaluations. HUD co-founders: Lorenss Martinsons (CPO) and Parth Patel (CTO). "
        "Education: Columbia University (computer science and physics). His prior experience is "
        "varied and ALL of the following are true (different sources surface different subsets; "
        "none contradict each other): Hume AI, AQR Capital Management, Chai Research, Quantedge, "
        "Standard Metrics, quantitative finance, and ML / LLM-interpretability research. Credit "
        "any of these as a correct prior role."
    ),
)
_research_jay.slug = "research-jay-ram"

tasks = [_web_research_rust, _research_jay]
