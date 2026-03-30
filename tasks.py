"""Task definitions for the deepresearch environment.

Each task is created via scenario.task() and can be run locally or remotely:

    python local_test.py --list
    python local_test.py --task research_ieee_2010
    python local_test.py --task multihop_marconi_advisor --model gpt-4o
"""

from env import (
    multi_hop_research,
    research,
    verify_claim,
)

# =============================================================================
# RESEARCH (binary: 1.0 if answer contains expected string, 0.0 otherwise)
# =============================================================================

# Easy: unique factual answer, single keyword
research_ieee_2010 = research.task(
    question="Who received the IEEE Frank Rosenblatt Award in 2010?",
    answer_includes="Michio Sugeno",
)
research_ieee_2010.slug = "research-ieee-2010"

# Medium: niche academic domain
research_jerlov_2018 = research.task(
    question="Who was awarded the Oceanography Society's Jerlov Award in 2018?",
    answer_includes="Annick Bricaud",
)
research_jerlov_2018.slug = "research-jerlov-2018"

# Hard: multiple acceptable answer forms, needs disambiguation
research_cambridge_college = research.task(
    question="What's the name of the women's liberal arts college in Cambridge, Massachusetts?",
    answer_includes=["Radcliffe College", "Radcliffe"],
)
research_cambridge_college.slug = "research-cambridge-college"

# =============================================================================
# VERIFY-CLAIM (binary: 1.0 if verdict matches, 0.0 otherwise)
# =============================================================================

# Easy: obviously true, widely known fact
verify_eiffel_paris = verify_claim.task(
    claim="The Eiffel Tower is located in Paris, France.",
    expected_verdict="true",
)
verify_eiffel_paris.slug = "verify-eiffel-paris"

# Medium: subtle factual error (Python was 1991, not 1995)
verify_python_1995 = verify_claim.task(
    claim="Python was created by Guido van Rossum in 1995.",
    expected_verdict="false",
)
verify_python_1995.slug = "verify-python-1995"

# Hard: requires nuanced research about history of geodetic surveys
verify_everest_tallest = verify_claim.task(
    claim="Mount Everest has always been recognized as the tallest mountain on Earth since its height was first measured.",
    expected_verdict="false",
)
verify_everest_tallest.slug = "verify-everest-tallest"

# =============================================================================
# MULTI-HOP-RESEARCH (partial credit: fraction of answer_parts found)
# =============================================================================

# Medium: two-hop — find a niche award winner, then find a detail about them
multihop_marconi_advisor = multi_hop_research.task(
    question="Who won the Marconi Prize in 2023, and who was their doctoral advisor?",
    answer_parts=["Hari Balakrishnan", "Randy Katz"],
)
multihop_marconi_advisor.slug = "multihop-marconi-advisor"

# Medium: three-step chain, with disambiguation
multihop_voyager_crs = multi_hop_research.task(
    question="Which institution built the Cosmic Ray Subsystem instrument aboard Voyager 1, who was the principal investigator of that instrument at launch, and at which university did that person earn their PhD?",
    answer_parts=[
        ["Caltech", "California Institute of Technology", "JPL", "Jet Propulsion Laboratory", "NASA"],
        ["Edward Stone", "Edward C. Stone", "Ed Stone"],
        ["University of Chicago", "UChicago"],
    ],
)
multihop_voyager_crs.slug = "multihop-voyager-crs"

# =============================================================================
# ALL_TASKS: canonical registry for discovery
# =============================================================================

ALL_TASKS = {
    # research (binary)
    "research_ieee_2010": research_ieee_2010,
    "research_jerlov_2018": research_jerlov_2018,
    "research_cambridge_college": research_cambridge_college,
    # verify-claim (binary)
    "verify_eiffel_paris": verify_eiffel_paris,
    "verify_python_1995": verify_python_1995,
    "verify_everest_tallest": verify_everest_tallest,
    # multi-hop-research (partial credit)
    "multihop_marconi_advisor": multihop_marconi_advisor,
    "multihop_voyager_crs": multihop_voyager_crs,
}
