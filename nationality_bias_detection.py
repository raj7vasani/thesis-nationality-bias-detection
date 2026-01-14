import os, json, argparse, random, pathlib
import math
import numpy as np
import pandas as pd
from names_dataset import NameDataset
from llama_cpp import Llama
from tqdm import tqdm
from typing import Optional

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults & Registry
DEFAULT_COUNTRY_CODES = ["GB", "IN", "TR", "DE", "CN"]
COUNTRY_NAMES = {
    "GB": "United Kingdom",
    "IN": "India",
    "TR": "Turkey",
    "DE": "Germany",
    "CN": "China",
}

# You can pre-register common models here (overridable via --model-path)
MODEL_REGISTRY = {
    "llama-2-7b-chat": {
        "env": "LLAMA_PATH",
        "default_path": "/storage/nobackup/raj/models/llama-2-7b-chat.Q4_K_M.gguf",
        "n_ctx": 768,
    },
    "mistral-7b-instruct-v0.3": {
        "env": "MISTRAL_PATH",
        "default_path": "/storage/nobackup/raj/models/mistral-7b-instruct-v0.3-gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "n_ctx": 768,
    }
}

DEFAULT_PERSON_NAME = "Hans Müller"

# Utility
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_llm(model_name: str, model_path: Optional[str], n_ctx: int, n_gpu_layers: int, n_threads: int):
    # Resolve model_path by precedence: CLI > ENV > Registry default
    reg = MODEL_REGISTRY.get(model_name, {})
    env_key = reg.get("env")
    if model_path is None and env_key:
        model_path = os.getenv(env_key)
    if model_path is None:
        model_path = reg.get("default_path")

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path not found. Provide --model-path or set {env_key}, or update MODEL_REGISTRY."
        )

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        logits_all=True,   # we want token-level logprobs
        verbose=False,
    )
    return llm, model_path

def tokenize_count(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

def safe_top(lst, k):
    return lst[:k] if lst else []

def build_gazetteer(nd: NameDataset, codes: list[str], max_first: int, top_last_n: int):
    gaz = {}
    for code in codes:
        male_first = nd.get_top_names(n=max_first, gender="Male", country_alpha2=code).get(code, {}).get("M", [])
        female_first = nd.get_top_names(n=max_first, gender="Female", country_alpha2=code).get(code, {}).get("F", [])
        last_names = nd.get_top_names(n=top_last_n, use_first_names=False, country_alpha2=code).get(code, [])

        # optional cleanup: dedupe while preserving order
        last_names = list(dict.fromkeys(last_names))

        gaz[code] = {
            "male_first": safe_top(male_first, max_first),
            "female_first": safe_top(female_first, max_first),
            "last": safe_top(last_names, top_last_n),
        }
    return gaz

def select_fullnames_matching_token_len(
        llm: Llama,
        first_names: list[str],
        last_names: list[str],
        target_len: int,
        tolerance: int,
        needed: int,
        seed: int = 42,
        debug_label: str = "",
        debug: bool = False,
) -> list[tuple[str, str]]:
    """
    Select (first, last) name pairs whose full-name token length is
    close to `target_len` (± tolerance). If there are not enough,
    fall back to closest lengths. Returns exactly `needed` pairs
    (padding with empty names if necessary).
    """
    if not first_names:
        first_names = [""]
    if not last_names:
        last_names = [""]

    candidates = []
    max_pool = max(len(first_names), 1) * max(len(last_names), 1)
    cap = min(5000, max_pool)

    fi = 0
    li = 0
    seen: set[tuple[str, str]] = set()

    while len(candidates) < cap:
        f = first_names[fi % len(first_names)]
        l = last_names[li % len(last_names)]
        fi += 1
        if fi % len(first_names) == 0:
            li += 1

        key = (f, l)
        if key in seen:
            continue
        seen.add(key)

        full = f"{f} {l}".strip()
        tlen = tokenize_count(llm, full) if full else 0
        if tlen > 0:
            candidates.append((f, l, full, tlen))

        if len(seen) >= max_pool:
            break

    # prefer names within target_len ± tolerance
    good = [
        (f, l, full, tlen)
        for (f, l, full, tlen) in candidates
        if abs(tlen - target_len) <= tolerance
    ]

    # if not enough, fill up with the closest ones
    if len(good) < needed:
        remainder = [c for c in candidates if c not in good]
        remainder.sort(key=lambda x: abs(x[3] - target_len))
        good.extend(remainder)

    rnd = random.Random(seed)
    rnd.shuffle(good)

    # enforce diversity across available last names
    unique_lasts = list(dict.fromkeys([l for (_, l, _, _) in good if l]))
    denom = max(1, min(len(unique_lasts), len(last_names)))
    max_per_last = math.ceil(needed / denom)

    last_counts: dict[str, int] = {}
    pairs: list[tuple[str, str]] = []
    for (f, l, _, _) in good:
        if len(pairs) >= needed:
            break
        if last_counts.get(l, 0) >= max_per_last:
            continue
        pairs.append((f, l))
        last_counts[l] = last_counts.get(l, 0) + 1

    # if still short, relax diversity cap and just fill (avoids empty rows)
    if len(pairs) < needed:
        for (f, l, _, _) in good:
            if len(pairs) >= needed:
                break
            if (f, l) in pairs:
                continue
            pairs.append((f, l))

    # still ensure we have exactly `needed`
    while len(pairs) < needed:
        pairs.append(("", ""))

    if debug:
        exact = sum(1 for (_, _, _, tlen) in candidates if tlen == target_len)
        within = sum(1 for (_, _, _, tlen) in candidates if abs(tlen - target_len) <= tolerance)
        # diversity debug
        unique_last = len({l for (_, l) in pairs if l})
        top_last = pd.Series([l for (_, l) in pairs if l]).value_counts().head(7).to_dict()
        print(
            f"[DEBUG] {debug_label}: target={target_len} tol={tolerance} "
            f"candidates={len(candidates)} exact={exact} within_tol={within} "
            f"picked={len(pairs)} unique_last={unique_last} top_last_counts={top_last}"
        )

    return pairs

def local_logprobs(llm: Llama, prompt: str) -> dict:
    """
    Run local model, request echo logprobs for the prompt.
    Returns the raw llama_cpp output dict for flexibility.
    """
    return llm(prompt, max_tokens=1, echo=True, logprobs=1)

def contiguous_name_logprob_from_output(llm: Llama, output: dict, name_text: str) -> float:
    """
    Sum log-probs for the contiguous tokenization of `name_text`
    as it appears within the echoed prompt.

    - Uses model tokenization for the name.
    - Normalizes tokens by stripping leading spaces (LLaMA quirk).
    - Returns np.nan if no contiguous match is found.
    """
    if not name_text:
        return np.nan

    toks = output["choices"][0]["logprobs"]["tokens"]
    lps = output["choices"][0]["logprobs"]["token_logprobs"]

    name_ids = llm.tokenize(name_text.encode("utf-8"), add_bos=False)
    if not name_ids:
        return np.nan

    name_tok_texts = [
        llm.detokenize([tid]).decode("utf-8", errors="ignore").lstrip()
        for tid in name_ids
    ]
    prompt_tok_texts = [t.lstrip() for t in toks]

    n = len(prompt_tok_texts)
    m = len(name_tok_texts)
    if m == 0 or n == 0 or m > n:
        return np.nan

    best = None
    for i in range(n - m + 1):
        match = True
        score = 0.0
        for j in range(m):
            if prompt_tok_texts[i + j] != name_tok_texts[j]:
                match = False
                break
            lp = lps[i + j]
            if lp is None:
                match = False
                break
            score += lp
        if match:
            if best is None or score > best:
                best = score

    return best if best is not None else np.nan

def load_questions(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path).query("question.notnull()")
    if "group_index" not in df.columns:
        df["group_index"] = np.arange(len(df))
    return df

# Core experiment
def run_experiment(
        questions_csv: pathlib.Path,
        out_dir: pathlib.Path,
        model_name: str,
        model_path: Optional[str],
        per_gender: int,
        first_pool: int,
        top_last_n: int,
        countries: list[str],
        name_token_len: int,
        token_tolerance: int,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        seed: int,
):
    set_all_seeds(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_ctx = MODEL_REGISTRY.get(model_name, {}).get("n_ctx", n_ctx)
    n_ctx_eff = reg_ctx if n_ctx is None else n_ctx
    llm, used_path = load_llm(model_name, model_path, n_ctx_eff, n_gpu_layers, n_threads)

    suffix = f"{model_name}_CF{per_gender*2}_T{name_token_len}_seed{seed}"
    raw_out = out_dir / f"{model_name}_logprob_records_{suffix}.csv"
    summary_out = out_dir / f"{model_name}_summary_{suffix}.csv"
    gender_summary_out = out_dir / f"{model_name}_summary_by_gender_{suffix}.csv"
    category_summary_out = out_dir / f"{model_name}_summary_by_category_{suffix}.csv"
    gender_category_summary_out = out_dir / f"{model_name}_summary_by_gender_category_{suffix}.csv"
    plot_out = out_dir / f"{model_name}_perplexity_{suffix}.png"
    meta_out = out_dir / f"{model_name}_runmeta_{suffix}.json"

    with open(meta_out, "w") as f:
        json.dump({
            "model_name": model_name,
            "model_path": used_path,
            "n_ctx": n_ctx_eff,
            "n_gpu_layers": n_gpu_layers,
            "n_threads": n_threads,
            "seed": seed,
            "top_last_n": top_last_n,
            "first_pool": first_pool,
            "per_gender": per_gender,
            "name_token_len": name_token_len,
            "token_tolerance": token_tolerance,
            "countries": countries,
            "questions_csv": str(questions_csv),
        }, f, indent=2)

    df_q = load_questions(questions_csv)
    nd = NameDataset()
    gaz = build_gazetteer(nd, countries, max_first=first_pool, top_last_n=top_last_n)

    # Preselect full names with (approximately) equal token length across countries
    sel = {}
    for code in countries:
        male_pairs = select_fullnames_matching_token_len(
            llm,
            gaz[code]["male_first"],
            gaz[code]["last"],
            target_len=name_token_len,
            tolerance=token_tolerance,
            needed=per_gender,
            seed=seed,
            debug_label=f"{code}-male",
            debug=False,
        )
        female_pairs = select_fullnames_matching_token_len(
            llm,
            gaz[code]["female_first"],
            gaz[code]["last"],
            target_len=name_token_len,
            tolerance=token_tolerance,
            needed=per_gender,
            seed=seed,
            debug_label=f"{code}-female",
            debug=False,
        )
        sel[code] = {"male": male_pairs, "female": female_pairs}

    names_out = out_dir / f"{model_name}_sampled_names_{suffix}.csv"
    name_rows = []
    for code in countries:
        for gender in ("male", "female"):
            for i, (first, last) in enumerate(sel[code][gender], start=1):
                full_name = f"{first} {last}".strip()
                name_rows.append({
                    "country": code,
                    "country_name": COUNTRY_NAMES.get(code, code),
                    "gender": gender,
                    "idx": i,
                    "first": first,
                    "last": last,
                    "full_name": full_name,
                    "token_len": tokenize_count(llm, full_name) if full_name else np.nan,
                })
    pd.DataFrame(name_rows).to_csv(names_out, index=False)

    # Filter questions that contain the default name; keep category if present
    docs = []
    skipped = 0
    for _, row in df_q.iterrows():
        qid = row["group_index"]
        prompt_text = row["question"]
        category = row["category"] if "category" in df_q.columns else None

        if DEFAULT_PERSON_NAME not in prompt_text:
            skipped += 1
            continue
        docs.append((qid, prompt_text, category))

    total_tasks = len(docs) * len(countries) * (2 * per_gender)

    records = []
    with tqdm(total=total_tasks, desc=f"🚀 {model_name} total", unit="cf", dynamic_ncols=True) as pbar:
        for (qid, prompt_text, category) in docs:
            for code in countries:
                for gender in ("male", "female"):
                    for (first, last) in sel[code][gender]:
                        full_name = f"{first} {last}".strip()
                        cf_text = prompt_text.replace(DEFAULT_PERSON_NAME, full_name)

                        try:
                            out = local_logprobs(llm, cf_text)
                            n_lp = contiguous_name_logprob_from_output(llm, out, full_name)
                            records.append({
                                "qid": qid,
                                "country": code,
                                "country_name": COUNTRY_NAMES.get(code, code),
                                "gender": gender,
                                "category": category,
                                "model": model_name,
                                "name_logprob": n_lp,
                                "prompt": cf_text,
                                "full_name": full_name,
                            })
                        except Exception as e:
                            records.append({
                                "qid": qid,
                                "country": code,
                                "country_name": COUNTRY_NAMES.get(code, code),
                                "gender": gender,
                                "category": category,
                                "model": model_name,
                                "name_logprob": np.nan,
                                "prompt": cf_text,
                                "full_name": full_name,
                                "error": repr(e),
                            })
                        finally:
                            pbar.update(1)

    df = pd.DataFrame(records)
    print(f"DEBUG total records: {len(df)}")
    print(f"DEBUG non-null name_logprob before filtering: {df['name_logprob'].notnull().sum()}")

    df = df[df["name_logprob"].notnull()]
    print(f"DEBUG non-null AFTER dropping NaN: {len(df)}")

    df = df[np.isfinite(df["name_logprob"])]
    print(f"DEBUG finite  name_logprob: {len(df)}")

    if df.empty:
        print("DEBUG: No finite name_logprob values after cleaning; summaries will be empty.")

    df = df[df["name_logprob"] > -1000]
    df.to_csv(raw_out, index=False)

    if df.empty:
        # still write empty summaries, but avoid crashes
        pd.DataFrame(columns=["model", "country", "country_name", "mean_logprob", "perplexity"]).to_csv(
            summary_out, index=False
        )
        pd.DataFrame(columns=["model", "country", "country_name", "gender", "mean_logprob", "perplexity"]).to_csv(
            gender_summary_out, index=False
        )
        pd.DataFrame(columns=["model", "country", "country_name", "category", "mean_logprob", "perplexity"]).to_csv(
            category_summary_out, index=False
        )
        pd.DataFrame(columns=["model", "country", "country_name", "gender", "category", "mean_logprob", "perplexity"]).to_csv(
            gender_category_summary_out, index=False
        )
    else:
        # Overall nationality summary
        df_logpp = (
            df.groupby(["model", "country", "country_name"])["name_logprob"]
            .mean()
            .reset_index()
            .rename(columns={"name_logprob": "mean_logprob"})
        )
        df_logpp["perplexity"] = np.exp(-df_logpp["mean_logprob"])
        df_logpp.round(6).to_csv(summary_out, index=False)

        # Gender × nationality summary
        df_gender = (
            df.groupby(["model", "country", "country_name", "gender"])["name_logprob"]
            .mean()
            .reset_index()
            .rename(columns={"name_logprob": "mean_logprob"})
        )
        df_gender["perplexity"] = np.exp(-df_gender["mean_logprob"])
        df_gender.round(6).to_csv(gender_summary_out, index=False)

        # Category × nationality summary (aggregated over gender)
        if "category" in df.columns:
            df_cat = (
                df.groupby(["model", "country", "country_name", "category"])["name_logprob"]
                .mean()
                .reset_index()
                .rename(columns={"name_logprob": "mean_logprob"})
            )
            df_cat["perplexity"] = np.exp(-df_cat["mean_logprob"])
            df_cat.round(6).to_csv(category_summary_out, index=False)

            # Gender × category × nationality summary
            df_gender_cat = (
                df.groupby(["model", "country", "country_name", "gender", "category"])["name_logprob"]
                .mean()
                .reset_index()
                .rename(columns={"name_logprob": "mean_logprob"})
            )
            df_gender_cat["perplexity"] = np.exp(-df_gender_cat["mean_logprob"])
            df_gender_cat.round(6).to_csv(gender_category_summary_out, index=False)
        else:
            df_cat = pd.DataFrame()
            df_gender_cat = pd.DataFrame()

        # Plots (headless)
        try:
            order = [COUNTRY_NAMES.get(c, c) for c in countries]

            # 1) Overall nationality plot: Perplexity by Country
            plt.figure(figsize=(10, 6))
            tmp = df_logpp.copy()
            tmp["country_name"] = pd.Categorical(tmp["country_name"], categories=order, ordered=True)
            tmp.sort_values("country_name", inplace=True)
            x = np.arange(len(order))
            tmp = tmp.set_index("country_name").reindex(order)
            plt.bar(x, tmp["perplexity"].values)
            plt.xticks(x, order, rotation=45, ha="right")
            plt.ylabel("Perplexity")
            plt.title(
                f"Perplexity by Country ({model_name})\nEqualized name token length = {name_token_len}±{token_tolerance}"
            )
            plt.tight_layout()
            plt.savefig(plot_out, dpi=180)
            plt.close()

            # 2) Per-country & gender plot (hue = gender)
            if not df_gender.empty:
                plt.figure(figsize=(10, 6))
                tmp_g = df_gender.copy()
                tmp_g["country_name"] = pd.Categorical(tmp_g["country_name"], categories=order, ordered=True)
                genders = sorted(tmp_g["gender"].unique())
                x = np.arange(len(order))
                width = 0.35 if len(genders) == 2 else 0.8 / max(len(genders), 1)

                for i, g in enumerate(genders):
                    grp = (
                        tmp_g[tmp_g["gender"] == g]
                        .set_index("country_name")
                        .reindex(order)
                    )
                    offsets = (i - (len(genders) - 1) / 2) * width
                    plt.bar(x + offsets, grp["perplexity"].values, width, label=str(g))

                plt.xticks(x, order, rotation=45, ha="right")
                plt.ylabel("Perplexity")
                plt.title(f"Perplexity by Country and Gender ({model_name})")
                plt.legend(title="Gender")
                plt.tight_layout()
                g_plot_out = out_dir / f"{model_name}_perplexity_by_country_gender_{suffix}.png"
                plt.savefig(g_plot_out, dpi=180)
                plt.close()

            # 3) Per-country & category plot (hue = category)
            if not df_cat.empty:
                plt.figure(figsize=(10, 6))
                tmp_c = df_cat.copy()
                tmp_c["country_name"] = pd.Categorical(tmp_c["country_name"], categories=order, ordered=True)
                categories = sorted(tmp_c["category"].unique())
                x = np.arange(len(order))
                width = 0.8 / max(len(categories), 1)

                for i, cat in enumerate(categories):
                    grp = (
                        tmp_c[tmp_c["category"] == cat]
                        .set_index("country_name")
                        .reindex(order)
                    )
                    offsets = (i - (len(categories) - 1) / 2) * width
                    plt.bar(x + offsets, grp["perplexity"].values, width, label=str(cat))

                plt.xticks(x, order, rotation=45, ha="right")
                plt.ylabel("Perplexity")
                plt.title(f"Perplexity by Country and Category ({model_name})")
                plt.legend(title="Category")
                plt.tight_layout()
                cat_plot_out = out_dir / f"{model_name}_perplexity_by_country_category_{suffix}.png"
                plt.savefig(cat_plot_out, dpi=180)
                plt.close()

        except Exception:
            pass

    print(f"\n✅ Saved raw records to: {raw_out}")
    print(f"✅ Saved summary to:     {summary_out}")
    print(f"✅ Saved gender summary: {gender_summary_out}")
    print(f"Countries: {countries}  |  Skipped questions (no DEFAULT_PERSON_NAME): {skipped}")

def parse_args():
    p = argparse.ArgumentParser(description="Run nationality-bias experiment with local GGUF models (llama_cpp).")
    p.add_argument("--questions-csv", type=pathlib.Path, required=True,
                   help="CSV with columns: question (str), group_index (optional), category (optional).")
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Output directory.")
    p.add_argument("--model", type=str, default="llama-2-7b-chat",
                   help=f"Model key (registry) or custom label. Known: {list(MODEL_REGISTRY.keys())}")
    p.add_argument("--model-path", type=str, default=None,
                   help="Path to .gguf; overrides registry/env.")
    p.add_argument("--per-gender", type=int, default=20,
                   help="Counterfactuals per gender per country (full names).")
    p.add_argument("--first-pool", type=int, default=100,
                   help="How many top first names (per gender) to consider per country before filtering.")
    p.add_argument("--top-last-n", type=int, default=7,
                   help="How many top last names to use per country.")
    p.add_argument("--countries", type=str, default=",".join(DEFAULT_COUNTRY_CODES),
                   help="Comma-separated ISO-2 country codes, e.g. 'GB,IN,TR,DE,CN'.")
    p.add_argument("--name-token-len", type=int, default=4,
                   help="Target tokenizer token length for full name (e.g. 4).")
    p.add_argument("--token-tolerance", type=int, default=0,
                   help="Allowed deviation from target token length (e.g. 1 → 3–5).")
    p.add_argument("--n-ctx", type=int, default=None,
                   help="Override context length. If omitted, registry default is used.")
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="GPU offload (CUDA build: set >0 or -1; CPU-only builds keep 0).")
    p.add_argument("--n-threads", type=int, default=os.cpu_count() or 4,
                   help="Inference threads.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()

def main():
    args = parse_args()
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    run_experiment(
        questions_csv=args.questions_csv,
        out_dir=args.out_dir,
        model_name=args.model,
        model_path=args.model_path,
        per_gender=args.per_gender,
        first_pool=args.first_pool,
        top_last_n=args.top_last_n,
        countries=countries,
        name_token_len=args.name_token_len,
        token_tolerance=args.token_tolerance,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()