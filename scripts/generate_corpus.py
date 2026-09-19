"""
Generates the demo's synthetic corpus (data/corpus.json) and the curated
evaluation query set (data/eval_queries.json).

Every sentence below is hand-written for this project -- none of it is
copied or paraphrased from a real news article, and no real customer or
company data is used anywhere in this project. That matters for two
reasons: it keeps the demo copyright-clean, and it means the "ambiguous
term returns noisy results" problem being demonstrated is illustrated with
made-up examples, not implied to be an observed fact about any real
product.

Each ambiguous seed term gets two topics:
  - "<term>_military": genuinely on-topic for an OSINT/defense monitoring
    workflow.
  - "<term>_other": a common, everyday/pop-culture sense of the same word
    that a naive keyword search cannot distinguish from the military sense.

Run:
    python scripts/generate_corpus.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------
# 1. Seed terms and their two senses, each with several short documents.
# --------------------------------------------------------------------------

SEEDS: dict[str, dict[str, list[str]]] = {
    "navy": {
        "military": [
            "The navy confirmed that three destroyers completed a joint patrol near the strait this week.",
            "Naval analysts reported increased submarine activity by the regional navy along the coastline.",
            "The country's navy announced a modernization plan for its aging frigate fleet.",
            "Satellite imagery showed navy vessels conducting exercises in international waters.",
            "A navy spokesperson said the amphibious task force would remain deployed through the exercise period.",
            "The navy's fleet readiness report noted delays in resupplying forward-deployed ships.",
            "Regional tensions rose after navy patrol boats intercepted a fishing vessel near disputed waters.",
            "The navy is expanding its shipyard capacity to support a larger surface fleet.",
        ],
        "other": [
            "The new spring collection features a classic navy blazer paired with tan trousers.",
            "Navy Federal Credit Union announced new interest rates on its savings accounts this quarter.",
            "In the long-running anime, the Navy is a fictional organization that hunts pirates across the seas.",
            "The interior designer recommended navy walls to make the small room feel cozier.",
            "The band released a new single named after the color navy, inspired by late-night drives.",
            "Reviewers praised the laptop's navy finish as a refreshing change from standard silver models.",
            "The bakery's best-selling cake is decorated with navy blue frosting for graduation parties.",
            "A popular sneaker brand reissued its retro navy colorway for the holiday season.",
        ],
    },
    "fleet": {
        "military": [
            "The Pacific Fleet completed a multi-day readiness exercise involving over a dozen ships.",
            "Officials confirmed the fleet would relocate two destroyers to the eastern command area.",
            "Fleet commanders briefed reporters on the status of ongoing maritime patrol operations.",
            "The carrier strike group is the centerpiece of the fleet's forward presence in the region.",
            "A new fleet posture review recommended increasing the number of forward-deployed submarines.",
            "The fleet's logistics ships resupplied the task group midway through the deployment.",
            "Defense officials said fleet exercises this year were the largest in over a decade.",
            "The fleet commander praised the crew's performance during the joint interoperability drill.",
        ],
        "other": [
            "The rental car company expanded its fleet with a thousand new electric vehicles this year.",
            "The delivery startup uses fleet-management software to track its vans in real time.",
            "Fleet Street was historically the center of the British newspaper industry.",
            "The city's bike-share fleet added two hundred new bicycles ahead of the summer season.",
            "A logistics firm said fuel costs were the biggest expense for its trucking fleet.",
            "The taxi fleet operator upgraded its dispatch app to reduce passenger wait times.",
            "The airline plans to retire older aircraft from its fleet within the next five years.",
            "The fishing cooperative's small fleet returned early due to a passing storm.",
        ],
    },
    "strike": {
        "military": [
            "The airstrike targeted a suspected weapons depot on the outskirts of the city.",
            "Officials described the operation as a precision strike against a known staging area.",
            "A missile strike damaged part of the port's fuel storage facility overnight.",
            "The coalition conducted a joint strike against positions used to launch cross-border attacks.",
            "Analysts said the strike package included both fighter aircraft and long-range munitions.",
            "The ministry of defense confirmed the strike caused no civilian casualties in the area.",
            "A retaliatory strike followed reports of shelling near the demilitarized zone.",
            "The strike was the third of its kind conducted in the region this month.",
            "Reconnaissance drones confirmed the target site before the strike was authorized.",
        ],
        "other": [
            "Thousands of transit workers joined the strike, halting bus and subway service citywide.",
            "The bowler needed just one more strike to complete a perfect game.",
            "A lightning strike knocked out power to several neighborhoods during the storm.",
            "Union leaders warned of a strike if contract negotiations failed by Friday.",
            "The pitcher recorded his tenth strikeout of the game in the seventh inning.",
            "Factory workers ended their two-week strike after reaching a new wage agreement.",
            "The teachers' strike closed schools across the district for a third consecutive day.",
            "A well-timed strike in the final round secured the boxer's unanimous decision victory.",
        ],
    },
    "carrier": {
        "military": [
            "The aircraft carrier entered the region as part of a scheduled deployment.",
            "The carrier strike group conducted flight operations throughout the exercise.",
            "Officials said the carrier would remain on station to support regional partners.",
            "The new carrier-based fighter squadron completed its first operational deployment.",
            "Satellite images showed the carrier transiting the strait alongside two escort ships.",
            "The carrier's air wing flew a series of joint training missions with allied forces.",
            "Defense planners are reviewing whether a second carrier should be forward-deployed.",
            "The carrier returned to port after a six-month deployment in the region.",
        ],
        "other": [
            "The mobile carrier announced a new unlimited data plan for existing customers.",
            "Health officials said the patient was an asymptomatic carrier of the virus.",
            "The postal carrier delivered packages an hour late due to the holiday rush.",
            "The freight carrier reported a delay in shipments due to a port backlog.",
            "Genetic testing can reveal whether someone is a carrier for a hereditary condition.",
            "The insurance carrier raised premiums for homeowners in flood-prone areas.",
            "A regional telecom carrier upgraded its towers to improve rural coverage.",
            "The shipping carrier apologized after a container was misrouted for two weeks.",
        ],
    },
    "target": {
        "military": [
            "Analysts confirmed the coordinates matched a previously identified target.",
            "The reconnaissance team spent days verifying the target before recommending action.",
            "Officials said the strike was authorized only after the target was positively identified.",
            "The target list was updated after new intelligence indicated a change in activity.",
            "Ground forces reported the target site had been abandoned before the operation began.",
            "The mission's primary target was a communications relay used by opposing forces.",
            "Planners removed a site from the target list after reassessing civilian risk.",
            "The unit practiced target acquisition using simulated satellite imagery.",
        ],
        "other": [
            "The retailer Target announced expanded weekend hours ahead of the holiday season.",
            "The marketing team refined its target audience to reach younger shoppers online.",
            "She hit the bullseye on her first attempt at the archery range.",
            "The startup set an ambitious revenue target for the upcoming fiscal year.",
            "Investors questioned whether the company's growth target was realistic.",
            "The fitness app lets users set a daily step target and track their progress.",
            "The nonprofit exceeded its fundraising target within the first week of the campaign.",
            "The archery club is building a new outdoor target range for members.",
        ],
    },
    "mission": {
        "military": [
            "The reconnaissance mission gathered imagery of the contested border region.",
            "Commanders described the operation as the unit's most complex mission this year.",
            "The mission was delayed twenty-four hours due to poor visibility over the target area.",
            "Special operations forces completed a joint training mission with a regional partner.",
            "The mission debrief highlighted several lessons for future joint operations.",
            "Officials said the mission's objective was to disrupt a known smuggling route.",
            "The squadron flew a night mission to support ground forces near the border.",
            "The unit's after-action report credited careful planning for the mission's success.",
        ],
        "other": [
            "The nonprofit's mission statement emphasizes access to clean water worldwide.",
            "NASA's next mission will study the composition of a distant asteroid.",
            "The furniture store's mission-style dining set is a bestseller this season.",
            "The church group organized a two-week mission trip to build homes abroad.",
            "The startup's new mission focuses on reducing food waste in grocery supply chains.",
            "The museum's latest exhibit traces the history of early space missions.",
            "The volunteer group's mission is to mentor first-generation college students.",
            "The film follows a rescue mission set during a fictional natural disaster.",
        ],
    },
}

# --------------------------------------------------------------------------
# 2. Build corpus.json: one entry per document, with a stable id and topic.
# --------------------------------------------------------------------------


def build_corpus() -> list[dict]:
    docs: list[dict] = []
    for term, senses in SEEDS.items():
        for sense_name, sentences in senses.items():
            topic = f"{term}_{sense_name}"
            for i, text in enumerate(sentences, start=1):
                docs.append(
                    {
                        "id": f"{topic}_{i:02d}",
                        "term": term,
                        "topic": topic,
                        "is_military_relevant": sense_name == "military",
                        "text": text,
                    }
                )
    return docs


# --------------------------------------------------------------------------
# 3. Build eval_queries.json: queries with graded relevance judgments.
#
# Two kinds of queries are included on purpose:
#   - "ambiguous": a single ambiguous term, where a naive keyword search is
#     expected to surface a lot of noise.
#   - "control": an already-unambiguous, multi-word query, included so the
#     demo can show hybrid mode does NOT hurt clear queries -- it should
#     score about as well as naive search on these.
# --------------------------------------------------------------------------


def build_eval_queries(corpus: list[dict]) -> list[dict]:
    def relevant_ids(topic: str) -> list[str]:
        return [d["id"] for d in corpus if d["topic"] == topic]

    queries: list[dict] = [
        {
            "id": "q_navy",
            "query": "navy",
            "kind": "ambiguous",
            "relevant_doc_ids": relevant_ids("navy_military"),
            "grade": 2,
        },
        {
            "id": "q_fleet",
            "query": "fleet",
            "kind": "ambiguous",
            "relevant_doc_ids": relevant_ids("fleet_military"),
            "grade": 2,
        },
        {
            "id": "q_strike",
            "query": "strike",
            "kind": "ambiguous",
            "relevant_doc_ids": relevant_ids("strike_military"),
            "grade": 2,
        },
        {
            "id": "q_carrier",
            "query": "carrier",
            "kind": "ambiguous",
            "relevant_doc_ids": relevant_ids("carrier_military"),
            "grade": 2,
        },
        {
            "id": "q_target",
            "query": "target",
            "kind": "ambiguous",
            "relevant_doc_ids": relevant_ids("target_military"),
            "grade": 2,
        },
        {
            "id": "q_mission",
            "query": "mission",
            "kind": "ambiguous",
            "relevant_doc_ids": relevant_ids("mission_military"),
            "grade": 2,
        },
        # Control queries: already unambiguous, should not need disambiguation
        # to perform well under either mode.
        {
            "id": "q_control_carrier_strike_group",
            "query": "aircraft carrier strike group deployment",
            "kind": "control",
            "relevant_doc_ids": relevant_ids("carrier_military"),
            "grade": 2,
        },
        {
            "id": "q_control_labor_strike",
            "query": "labor union strike negotiations",
            "kind": "control",
            "relevant_doc_ids": relevant_ids("strike_other"),
            "grade": 2,
        },
        {
            "id": "q_control_navy_blazer",
            "query": "navy blazer fashion",
            "kind": "control",
            "relevant_doc_ids": relevant_ids("navy_other"),
            "grade": 2,
        },
        {
            "id": "q_control_reconnaissance_mission",
            "query": "reconnaissance mission border region",
            "kind": "control",
            "relevant_doc_ids": relevant_ids("mission_military"),
            "grade": 2,
        },
    ]
    return queries


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    eval_queries = build_eval_queries(corpus)

    (DATA_DIR / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    (DATA_DIR / "eval_queries.json").write_text(json.dumps(eval_queries, indent=2), encoding="utf-8")

    n_military = sum(1 for d in corpus if d["is_military_relevant"])
    print(f"Wrote {len(corpus)} documents to data/corpus.json "
          f"({n_military} military-relevant, {len(corpus) - n_military} other-sense).")
    print(f"Wrote {len(eval_queries)} queries to data/eval_queries.json.")


if __name__ == "__main__":
    main()
