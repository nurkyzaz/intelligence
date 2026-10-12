"""Score the black-swan results in bits (the unit of the 'intelligence as compression' plan).

For every conflict puzzle the generator's answer is the RULE label. We compare three code lengths,
in bits, for that answer:
  model   : -log2 p_model(rule label)                  (the model's real next-token probabilities)
  ideal   : -log2 p_ideal(rule label)                  (Bayes learner that sees the same demos)
  uniform : log2 K                                     (guessing among the K labels)
and report the normalized gain G = (uniform - model) / (uniform - ideal), Step 3 of the plan.
The ideal learner assumes noiseless labels (the generator has none): it gives the rule 50% when the
demos cannot tell the features apart (m = 0) and 100% once the shortcut has a counterexample (m >= 1).

Run:  python bits_analysis.py data/blackswan_eval_scored.jsonl results
"""
import json, sys, os
import numpy as np
import matplotlib.pyplot as plt

LABELS = ["1", "2", "3"]
K = len(LABELS)
TIERS = ["nonsense", "natural", "hint"]


def ideal_p_rule(m):
    return 0.5 if m == 0 else 1.0


def bits_table(tasks):
    """tasks: E1 conflict puzzles with 'label_probs'. Returns {tier: {m: summary}}."""
    out = {}
    for tier in TIERS:
        out[tier] = {}
        for m in sorted({t["m"] for t in tasks}):
            ts = [t for t in tasks if t["tier"] == tier and t["m"] == m]
            if not ts:
                continue
            p = np.array([t["label_probs"] for t in ts])
            i_r = np.array([LABELS.index(t["rule_label"]) for t in ts])
            i_s = np.array([LABELS.index(t["short_label"]) for t in ts])
            p_rule, p_short = p[np.arange(len(ts)), i_r], p[np.arange(len(ts)), i_s]
            L_model = float(np.mean(-np.log2(np.clip(p_rule, 1e-12, 1))))
            L_ideal = abs(np.log2(ideal_p_rule(m)))
            L_unif = np.log2(K)
            out[tier][m] = dict(
                n=len(ts),
                bits_model=L_model, bits_ideal=float(L_ideal), bits_uniform=float(L_unif),
                bits_wasted=L_model - L_ideal,
                gain=float(max(0.0, (L_unif - L_model) / (L_unif - L_ideal))),
                p_rule=float(p_rule.mean()), p_shortcut=float(p_short.mean()),
                p_third=float((1 - p_rule - p_short).mean()),
                picks_shortcut=float(np.mean(p.argmax(1) == i_s)),
            )
    return out


def plot(tab, path):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ms = sorted(tab[TIERS[0]])
    for tier, c in zip(TIERS, ["C0", "C1", "C2"]):
        axs[0].plot(ms, [tab[tier][m]["bits_model"] for m in ms], "o-", color=c, label=f"Qwen3-4B · {tier}")
        axs[1].plot(ms, [tab[tier][m]["gain"] for m in ms], "o-", color=c, label=tier)
    axs[0].plot(ms, [tab[TIERS[0]][m]["bits_ideal"] for m in ms], "k--", label="ideal learner")
    axs[0].axhline(np.log2(K), color="gray", lw=0.8, ls=":", label="uniform guess")
    axs[0].set_xlabel("m = counterexamples to the shortcut"); axs[0].set_ylabel("bits to encode the right answer")
    axs[0].set_title("Cost in bits (lower is better)"); axs[0].legend(fontsize=7); axs[0].set_xticks(ms)
    axs[1].axhline(1, color="k", ls="--", lw=1, label="ideal learner")
    axs[1].set_ylim(0, 1.05); axs[1].set_xticks(ms)
    axs[1].set_xlabel("m = counterexamples to the shortcut"); axs[1].set_ylabel("normalized gain G")
    axs[1].set_title("Share of the learnable structure captured"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160)


if __name__ == "__main__":
    src, outdir = sys.argv[1], sys.argv[2]
    tasks = [json.loads(l) for l in open(src)]
    tasks = [t for t in tasks if t["split"] == "E1_blackswan"]
    tab = bits_table(tasks)
    for tier in TIERS:
        for m, r in tab[tier].items():
            print(f"{tier:9s} m={m}: model {r['bits_model']:.2f} bits | ideal {r['bits_ideal']:.2f} | "
                  f"wasted {r['bits_wasted']:.2f} | gain {r['gain']:.2f} | p(rule) {r['p_rule']:.2f} "
                  f"p(shortcut) {r['p_shortcut']:.2f} p(other) {r['p_third']:.2f} | picks shortcut {r['picks_shortcut']:.2f}")
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    json.dump({t: {str(m): v for m, v in d.items()} for t, d in tab.items()},
              open(f"{outdir}/bits_results.json", "w"), indent=2)
    plot(tab, f"{outdir}/figures/e1_bits.png")
    print("saved", f"{outdir}/bits_results.json", f"{outdir}/figures/e1_bits.png")
