"""Audit the licence of every dataset behind the task registry, before any weights are published.

    python scripts/licence_audit.py                 # query the Hub and write selectia/data/task_licences.json
    python scripts/licence_audit.py --offline       # rebuild the table from the committed json only

Why this exists: `Tobi-Bueck/customer-support-tickets` (the `support_tickets` task) is CC-BY-NC-4.0, so a
model trained on the `core` mixture inherits a non-commercial restriction. That is the same trap
`Luni/laya-jev-benchmark` documents for its own checkpoint. The released weights are only as free as the
most restricted dataset they were trained on, so the licence of every repo is recorded next to the
registry instead of assumed.

The per-task repo map is written out here rather than parsed from the loader source: the loaders call
`load_dataset` with a variable, and a task may read a raw file (`trec_fine`) or a local artifact
(`mario`, `synth`, `games`) rather than a Hub dataset. `None` means there is no external dataset.
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "selectia/data/task_licences.json"

# task -> Hugging Face dataset repo (None = synthetic / local / no external data)
TASK_REPOS = {
    "clinc_oos": "clinc/clinc_oos",
    "banking77": "mteb/banking77",
    "massive_intent": "SetFit/amazon_massive_intent_en-US",
    "massive_scenario": "SetFit/amazon_massive_scenario_en-US",
    "bitext_support": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    "support_tickets": "Tobi-Bueck/customer-support-tickets",
    "ag_news": "fancyzhx/ag_news",
    "dbpedia": "fancyzhx/dbpedia_14",
    "yahoo_topics": "community-datasets/yahoo_answers_topics",
    "newsgroups": "SetFit/20_newsgroups",
    "bbc_news": "SetFit/bbc-news",
    "trec": "SetFit/TREC-QC",
    "student_questions": "SetFit/student-question-categories",
    "dolly_category": "argilla/databricks-dolly-15k-curated-en",
    "imdb": "stanfordnlp/imdb",
    "sst2": "stanfordnlp/sst2",
    "sst5": "SetFit/sst5",
    "yelp": "Yelp/yelp_review_full",
    "amazon_stars": "SetFit/amazon_reviews_multi_en",
    "emotion": "dair-ai/emotion",
    "go_emotions": "google-research-datasets/go_emotions",
    "tweet_sentiment": "cardiffnlp/tweet_eval",
    "tweet_emotion": "cardiffnlp/tweet_eval",
    "tweet_irony": "cardiffnlp/tweet_eval",
    "tweet_offensive": "cardiffnlp/tweet_eval",
    "tweet_hate": "cardiffnlp/tweet_eval",
    "fin_sentiment": "zeroshot/twitter-financial-news-sentiment",
    "cr_reviews": "SetFit/SentEval-CR",
    "counterfactual": "SetFit/amazon_counterfactual_en",
    "subjectivity": "SetFit/subj",
    "hate_offensive": "SetFit/hate_speech_offensive",
    "civil_comments": "google/civil_comments",
    "toxic_chat": "lmsys/toxic-chat",
    "sms_spam": "ucirvine/sms_spam",
    "enron_spam": "SetFit/enron_spam",
    "insincere_questions": "SetFit/insincere-questions",
    "ade": "SetFit/ade_corpus_v2_classification",
    "snli": "stanfordnlp/snli",
    "mnli": "nyu-mll/multi_nli",
    "rte": "nyu-mll/glue",
    "qnli": "nyu-mll/glue",
    "qqp": "nyu-mll/glue",
    "mrpc": "nyu-mll/glue",
    "cola": "nyu-mll/glue",
    "paws": "google-research-datasets/paws",
    "boolq": "google/boolq",
    "strategyqa": "ChilleD/StrategyQA",
    "pubmedqa": "qiaojin/PubMedQA",
    "arc": "allenai/ai2_arc",
    "commonsense_qa": "tau/commonsense_qa",
    "qasc": "allenai/qasc",
    "openbookqa": "allenai/openbookqa",
    "sciq": "allenai/sciq",
    "hellaswag": "Rowan/hellaswag",
    "piqa": "ybisk/piqa",
    "social_iqa": "allenai/social_i_qa",
    "winogrande": "allenai/winogrande",
    "race": "ehovy/race",
    "mmlu": "cais/mmlu",
    "medqa": "GBaker/MedQA-USMLE-4-options",
    "truthfulqa": "truthfulqa/truthful_qa",
    "fin_phrasebank": "AdaptLLM/finance-tasks",
    "bias_in_bios": "LabHC/bias_in_bios",
    "helpsteer2": "nvidia/HelpSteer2",
    "helpsteer3_pref": "nvidia/HelpSteer3",
    "stsb": "SetFit/stsb",
    "hate_speech_scales": "ucberkeley-dlab/measuring-hate-speech",
    "liar2": "chengxuphd/liar2",
    "prosocial_safety": "allenai/prosocial-dialog",
    "ultrafeedback_pref": "HuggingFaceH4/ultrafeedback_binarized",
    "shp": "stanfordnlp/shp",
    "hh_rlhf": "Anthropic/hh-rlhf",
    "arena_pref": "lmarena-ai/arena-human-preference-55k",
    "reward_bench": "allenai/reward-bench",
    "glaive_tools": "glaiveai/glaive-function-calling-v2",
    "toolace": "Team-ACE/ToolACE",
    "hermes_tools": "NousResearch/hermes-function-calling-v1",
    "copa": "aps/super_glue",
    "wic": "aps/super_glue",
    "multirc": "aps/super_glue",
    "cb": "aps/super_glue",
    "fever": "copenlu/fever_gold_evidence",
    "wiki_qa": "microsoft/wiki_qa",
    "msmarco_rel": "microsoft/ms_marco",
    "medmcqa": "openlifescienceai/medmcqa",
    "quality": "emozilla/quality",
    "xstory_cloze": "juletxara/xstory_cloze",
    "agenttraj": "AgentGym/AgentTraj-L",
    "mind2web": "osunlp/Mind2Web",
    "hwu64": "FastFit/hwu_64",
    "dbpedia_l2": "DeveloperOats/DBPedia_Classes",
    "dbpedia_l3": "DeveloperOats/DBPedia_Classes",
    "trec_fine": "SetFit/TREC-QC",          # read as a raw file, not via the Hub loader
    "quality_full": "emozilla/quality",
    # no external dataset:
    "mario": None, "games": None, "synth": None, "abstain_probe": None, "offtopic_probe": None,
}

NONCOMMERCIAL = re.compile(r"(nc|non.?commercial|research.only|non.?commercial)", re.I)
COPYLEFT = re.compile(r"(gpl|agpl|lgpl|cc-by-sa|odbl|sspl)", re.I)
PERMISSIVE = re.compile(r"^(mit|apache-2\.0|bsd|cc0-1\.0|cc-by-[34]\.0|cc-by-4\.0|odc-by|unlicense|wtfpl|isc|mpl-2\.0|epl-2\.0)", re.I)


def repo_licence(repo, api):
    """Return (licence, source). card_data first, then the README YAML front matter, then the tags."""
    try:
        info = api.dataset_info(repo)
    except Exception as e:
        return None, f"error:{type(e).__name__}"
    cd = getattr(info, "card_data", None)
    lic = getattr(cd, "license", None)
    lic = ", ".join(lic) if isinstance(lic, (list, tuple)) else lic
    if lic:
        return str(lic), "card_data"
    tags = [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")]
    if tags:
        return ", ".join(tags), "tags"
    try:                                                    # last resort: the raw README front matter
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, "README.md", repo_type="dataset")
        m = re.search(r"^license\s*:\s*(.+)$", open(p, encoding="utf-8").read()[:4000], re.M)
        if m:
            return m.group(1).strip().strip("'\""), "readme"
    except Exception:
        pass
    return None, "none"


def classify(lic):
    if not lic:
        return "unknown"
    if NONCOMMERCIAL.search(lic):
        return "noncommercial"
    if COPYLEFT.search(lic):
        return "copyleft"
    if PERMISSIVE.match(lic.strip()):
        return "commercial_ok"
    if lic.strip().lower() in ("other", "unknown"):
        return "unknown"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--offline", action="store_true", help="do not touch the Hub; reprint the committed table")
    a = ap.parse_args()

    if a.offline:
        table = json.load(open(a.out))
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        cache = {}
        for repo in sorted({r for r in TASK_REPOS.values() if r}):
            cache[repo] = repo_licence(repo, api)
            print(f"[licence] {repo:60s} {cache[repo][0] or '(none)':12s} via {cache[repo][1]}", flush=True)
        from selectia import data as D
        table = {}
        for task, meta in D.TASKS.items():
            repo = TASK_REPOS.get(task)
            lic, src = cache[repo] if repo else (None, "no_external_dataset")
            table[task] = dict(repo=repo, licence=lic, source=src, class_=classify(lic),
                               heldout=bool(meta["heldout"]), trained=not meta["heldout"])
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(table, open(a.out, "w"), indent=1, sort_keys=True)

    by_class = {}
    for t, r in table.items():
        by_class.setdefault(r["class_"], []).append(t)
    print(f"\n[audit] {len(table)} tasks; classes:")
    for k in sorted(by_class):
        print(f"[audit]   {k:14s} {len(by_class[k]):3d}  {', '.join(sorted(by_class[k]))}")
    offenders = [t for t, r in table.items() if r["class_"] == "noncommercial"]
    trained_offenders = [t for t in offenders if table[t]["trained"]]
    print(f"\n[audit] non-commercial: {offenders}")
    print(f"[audit] trained on (blocks a permissive weight release): {trained_offenders}")
    unknown = [t for t, r in table.items() if r["class_"] == "unknown" and r["repo"]]
    print(f"[audit] licence not declared on the Hub (needs a manual read): {len(unknown)} tasks")
    for t in sorted(unknown):
        print(f"          {t:24s} {table[t]['repo']}")


if __name__ == "__main__":
    main()
