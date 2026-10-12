# Intelligence as Efficient Compression — MVP Research Plan

Oct 12, 2026 · @nnur

## Summary and research questions

This project defines intelligence as resource-bounded, sample-efficient predictive compression across many environments, and tests that definition against Legg–Hutter and Chollet. The MVP takes about six weeks and produces a formal note with five propositions plus a small reproducible benchmark.

**Working definition.** Intelligence is the ability to discover, from limited data and compute, short descriptions of unfamiliar environments that support prediction (phase 1) and intervention (phase 2).

1. **RQ1 (conceptual).** Which definition best satisfies the desiderata in section 5 on the test cases in section 6?
2. **RQ2 (formal).** Which desiderata does the proposed measure provably satisfy, and exactly where does it diverge from Legg–Hutter?
3. **RQ3 (empirical).** On simplicity-weighted toy environments, do rankings match the case verdicts, and do they survive a change of reference language?

Scope: prediction first. The interventional extension and action-based environments come in phase 2.

## Setup and notation

An agent is scored on sequences drawn from computable environments, with environments weighted by how short their description is in a fixed language.

```math
\begin{aligned}
&\mathcal{X} && \text{finite observation alphabet (MVP: at most 4 symbols)}\\
&\mu && \text{environment: a computable measure over sequences } x_1 x_2 \ldots \in \mathcal{X}^{\infty}\\
&\mathcal{L} && \text{fixed reference description language (DSL)}\\
&\ell(\mu) && \text{bits of the shortest } \mathcal{L}\text{-program that generates } \mu \text{ (computable proxy for } K_U(\mu)\text{)}\\
&\xi(\mu) = 2^{-\ell(\mu)} && \text{simplicity prior over environments}\\
&A && \text{agent: maps a history } x_{1:n} \text{ to a predictive distribution } p_A(\cdot \mid x_{1:n})\\
&n,\ m && \text{samples seen before predicting; length of the held-out prediction block}\\
&|A|,\ T_A && \text{agent size in bits (code + parameters); compute steps used}
\end{aligned}
```

Everything below is defined relative to the language L. Section 8 tests how much that choice matters, because Leike and Hutter (2015) showed the choice of reference machine can distort Legg–Hutter scores arbitrarily.

## Three definitions, formalized

The three definitions differ in one core quantity: reward (Legg–Hutter), skill-acquisition efficiency (Chollet), or predictive compression under a compute budget (proposed).

**Legg–Hutter (2007).** Intelligence is expected reward across all computable environments, weighted by simplicity.

```math
\Upsilon^{\mathrm{LH}}(\pi) = \sum_{\mu \in \mathcal{E}} 2^{-K_U(\mu)}\, V^{\pi}_{\mu}, \qquad V^{\pi}_{\mu} = \mathbb{E}^{\pi}_{\mu}\Big[\sum_{t=1}^{\infty} r_t\Big] \le 1
```

**Chollet (2019), schematic.** Intelligence is how efficiently skill is acquired, relative to the priors and experience used. The expression below simplifies his algorithmic-information formula; it is not his exact definition.

```math
I^{\mathrm{C}}(A) \approx \mathbb{E}_{T \in \mathcal{S}}\left[ \frac{\theta_T \cdot \mathrm{GD}_T}{P_T + E_T} \right]
```

For each task T in scope S: theta is the skill level reached, GD the generalization difficulty, P and E the priors and experience used, all in bits.

**Proposed (this project).** Intelligence is the simplicity-weighted, learning-curve-averaged gain in predictive compression, compared among agents within a compute budget b.

```math
\Upsilon^{\mathrm{N}}_{b}(A) = \sum_{\mu \in \mathcal{E}} \xi(\mu)\, \bar{G}_A(\mu) \qquad \text{for agents with } c(A) \le b
```

| Definition | Core quantity | Generality comes from | Resources | Computable |
| --- | --- | --- | --- | --- |
| Legg–Hutter | Expected reward | Sum over all computable environments | Ignored | No |
| Chollet | Skill-acquisition efficiency | Scope of tasks | Central: priors + experience | Approximated by benchmarks (ARC) |
| Proposed | Predictive compression gain | Sum over sampled environments | Central: budget classes | Yes, given a DSL prior |

## The proposed measure, step by step

The measure turns your three components (compress, predict, be efficient) into one number per environment, then averages over environments. Compression and prediction merge into a single quantity, because encoding an outcome predicted with probability p costs exactly −log2 p bits.

```mermaid
flowchart LR
    A["Sample a world μ ~ ξ"] --> B["Agent sees n samples"]
    B --> C["Predicts next m: code length L_A"]
    C --> D["Normalize between uniform and true: gain G"]
    D --> E["Repeat for n = 1, 2, 4, ...: learning-curve gain"]
    E --> F["Average over worlds"]
    F --> G{"Cost c(A) ≤ b?"}
    G -->|yes| H["Score within budget class"]
```

Read it left to right: a learning curve of predictions gives one gain per world, worlds are averaged, and agents compete only inside their cost budget.

**Step 1: predictive code length.** The agent sees n samples, then predicts the next m. Its cost is the expected number of bits needed to encode that held-out block.

```math
L_A(\mu; n) = \mathbb{E}_{x \sim \mu}\left[ -\log_2 p_A\!\left(x_{n+1:n+m} \mid x_{1:n}\right) \right]
```

**Step 2: two anchors.** A uniform guesser costs the most; the true environment itself costs the least.

```math
L_{\mathrm{unif}} = m \log_2 |\mathcal{X}|, \qquad L^{*}(\mu; n) = \mathbb{E}_{x \sim \mu}\left[ -\log_2 \mu\!\left(x_{n+1:n+m} \mid x_{1:n}\right) \right]
```

**Step 3: normalized gain.** The share of the learnable structure the agent captured: 0 means no better than guessing, 1 means as good as the true environment.

```math
G_A(\mu; n) = \max\!\left(0,\; \frac{L_{\mathrm{unif}} - L_A(\mu; n)}{L_{\mathrm{unif}} - L^{*}(\mu; n)} \right) \in [0, 1]
```

Pure-noise environments (where the true cost equals the uniform cost) have no structure to find and get weight 0.

**Step 4: sample efficiency.** Average the gain over a learning curve with data doubling each step, halving the weight each time. Finding structure from little data counts most.

```math
\bar{G}_A(\mu) = \sum_{j=0}^{J} w_j\, G_A\!\left(\mu; 2^{j}\right), \qquad w_j = \frac{2^{-j}}{\sum_{k=0}^{J} 2^{-k}}
```

**Step 5: resource cost.** Size and compute are measured in the same unit, bits, following Levin's time-bounded complexity.

```math
c(A) = |A| + \log_2 T_A
```

The MVP compares agents only within a budget class (c at most b) and reports the frontier of score against cost. This avoids an arbitrary trade-off constant between bits of model and bits of prediction.

**Step 6: estimate by sampling.** Draw fresh environments from the prior at every evaluation, so no fixed lookup table can memorize the test set.

```math
\hat{\Upsilon}^{\mathrm{N}}(A) = \frac{1}{N} \sum_{i=1}^{N} \bar{G}_A(\mu_i), \qquad \mu_i \sim \xi
```

**Step 7 (phase 2): interventional compression.** This is where "useful" gets a precise meaning. The environment becomes a structural causal model M; the agent learns from n observational samples, then predicts outcomes under held-out interventions do(X = x) drawn from a query set Q.

```math
L^{\mathrm{do}}_A(\mathcal{M}; n) = \mathbb{E}_{\mathrm{do}(X=x) \sim \mathcal{Q}}\; \mathbb{E}_{v \sim P_{\mathcal{M}}(\cdot \mid \mathrm{do}(X=x))}\left[ -\log_2 p_A\!\left(v \mid \mathrm{do}(X=x), \mathcal{D}_n\right) \right]
```

The interventional gain is defined exactly as in Steps 3–4, then mixed with the observational gain.

```math
\bar{G}^{\alpha}_A = (1-\alpha)\, \bar{G}^{\mathrm{obs}}_A + \alpha\, \bar{G}^{\mathrm{do}}_A, \qquad \alpha \in [0, 1]
```

The weight alpha is a free choice. The MVP reports alpha = 0 and alpha = 1 separately rather than picking a mixture.

## Desiderata

Eight desiderata act as the axioms a good definition should satisfy; the proposed measure meets seven by construction or proof (D7 only in phase 2) and leaves D8 open for the experiment.

1. **D1 Normalization.** A uniform guesser scores 0; an agent that knows the true environment scores 1.
2. **D2 Monotonicity.** If agent A's loss is no worse than B's on every environment and every n, A scores at least as high.
3. **D3 Generality.** An agent that excels on one environment class and guesses elsewhere scores at most that class's prior mass.
4. **D4 Anti-memorization.** No fixed, finite lookup table scores high in expectation.
5. **D5 Sample efficiency.** Reaching the same loss from fewer samples scores higher.
6. **D6 Resource sensitivity.** The same predictions from a smaller or faster agent are preferred.
7. **D7 Causal sensitivity.** Agents with equal observational loss but different interventional loss are distinguished.
8. **D8 Reference robustness.** Rankings stay stable when the description language L changes.

| Desideratum | Legg–Hutter | Chollet | Proposed |
| --- | --- | --- | --- |
| D1 Normalization | Yes, bounded in \[0, 1\] | Depends on scaling | Yes (P1) |
| D2 Monotonicity in loss | Not applicable (reward) | Not applicable | Yes, by construction |
| D3 Generality | Yes | Yes, via task scope | Yes (P5) |
| D4 Anti-memorization | Yes | Yes, via private test set | Yes, via fresh sampling |
| D5 Sample efficiency | No | Yes, central | Yes (Step 4) |
| D6 Resource sensitivity | No | Partial: priors, not compute | Yes (Step 5) |
| D7 Causal sensitivity | Only if rewards depend on interventions | Not explicit | Yes when alpha > 0 (P4) |
| D8 Reference robustness | No (Leike–Hutter 2015) | Open | Open: tested in H3 |

The Legg–Hutter and Chollet columns are my assessment and should be checked against the original papers before submission.

## Test cases

Seven thought experiments separate the definitions; the Oracle is the one case where they genuinely disagree about what intelligence is.

| Case | What it is | Legg–Hutter | Chollet | Proposed (alpha = 0) | Intuition |
| --- | --- | --- | --- | --- | --- |
| Oracle | Learns to predict perfectly but never acts | 0: earns no reward | High if it learns efficiently | Near 1 | Contested: understanding vs achieving |
| Bacterium | Fixed reflexes that collect reward, no model | Small but positive | Near 0: acquires nothing | Near 0 | Low |
| Specialist (JPEG) | Hand-built compressor for one data class | Low | Low: skill came from the designer | At most the class's prior mass (P5) | Low |
| Lookup table | Memorized answers to a fixed benchmark | Low | 0 | Near 0 under fresh sampling; huge cost | Low |
| gzip predictor | General-purpose but shallow compressor | Low to moderate | Low | Low to moderate, very cheap | Low, nonzero |
| Solomonoff / AIXI | Idealized universal predictor / agent | AIXI is optimal by construction | Undefined: infinite priors | Near 1 but infinite cost: outside every budget class | Ideal, not real |
| Human or LLM | Real general learner | High | High | High | High |

The Solomonoff row shows why budgets matter: without them, the proposed measure is maximized by an incomputable agent, exactly as Legg–Hutter is maximized by AIXI (P2).

## Propositions

Five propositions pin down what the measure does; P1, P4 and P5 are elementary, P2 restates known Solomonoff results, and P3 is a construction. All proofs below are sketches to write out in Phase 0.

### P1. Range and optimum

For every agent, environment and n, the gain lies in \[0, 1\], and equals 1 exactly when the agent's prediction matches the true conditional on the held-out block. The proof is one line: cross-entropy splits into entropy plus a KL divergence.

```math
L_A(\mu; n) = L^{*}(\mu; n) + \mathbb{E}_{x_{1:n}}\Big[ D_{\mathrm{KL}}\big(\mu(\cdot \mid x_{1:n}) \,\|\, p_A(\cdot \mid x_{1:n})\big) \Big] \ge L^{*}(\mu; n)
```

KL is non-negative, and zero only when the two distributions are equal (Gibbs' inequality). So the numerator of G never exceeds its denominator.

### P2. Without budgets, universal prediction wins

Solomonoff's universal mixture M dominates every computable environment, which bounds its total excess code length by the environment's complexity.

```math
M(x_{1:n}) \ge 2^{-K_U(\mu) - O(1)}\, \mu(x_{1:n}) \quad \Longrightarrow \quad \mathbb{E}_{\mu}\big[ \log_2 \mu(x_{1:n}) - \log_2 M(x_{1:n}) \big] \le K_U(\mu) + O(1) \quad \text{for all } n
```

The bounded total forces the per-step excess to zero, so the gain of M tends to 1 in every structured environment. Two consequences:

- **Simplicity and sample efficiency are linked exactly:** an environment of K bits costs a universal learner at most about K extra bits in total.
- **Budgets are what make the measure non-trivial:** M is incomputable, so its cost is infinite and it sits outside every budget class, just as AIXI is unbeatable but unbuildable under Legg–Hutter.

### P3. Where the proposed measure and Legg–Hutter agree and diverge

**Agreement.** In prediction environments that pay reward as an increasing affine function of the agent's log-score (suitably bounded), both measures rank agents by simplicity-weighted predictive code length. They differ only in the normalization (Step 3) and learning-curve weights (Step 4).

**Divergence.** Treat an agent as a pair (predictor, policy). Let A pair a Bayes-optimal predictor with a do-nothing policy, and B pair a uniform predictor with a reward-optimal reflex. On environments where reward depends only on actions, the proposed measure ranks A above B (near 1 vs 0) and Legg–Hutter ranks B above A. The measures split exactly where reward is not prediction: the Oracle case.

### P4. Observation alone cannot certify causal understanding

Take the linear-Gaussian model below with a nonzero. Agent A uses the true graph X → Y; agent B uses the reversed graph Y → X; both fit the same joint distribution exactly.

```math
X \sim \mathcal{N}(0, 1), \qquad Y = aX + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2), \qquad a \neq 0
```

Same joint means identical observational loss, so the observational gains are equal. Under an intervention on X the agents disagree, because B's model says intervening on X cannot move Y.

```math
\text{truth: } Y \mid \mathrm{do}(X{=}x) \sim \mathcal{N}(ax,\ \sigma^2), \qquad \text{agent B: } Y \mid \mathrm{do}(X{=}x) \sim \mathcal{N}(0,\ a^2+\sigma^2)
```

B's excess cost, in nats, is:

```math
D_{\mathrm{KL}}\big(\mathcal{N}(ax, \sigma^2) \,\|\, \mathcal{N}(0, a^2+\sigma^2)\big) = \tfrac{1}{2}\ln\frac{a^2+\sigma^2}{\sigma^2} + \frac{\sigma^2 + a^2x^2}{2(a^2+\sigma^2)} - \tfrac{1}{2}
```

At x = 0 the excess equals the bracketed term below, which is positive; any nonzero x only adds to it.

```math
\ln(1+r) > \frac{r}{1+r} = 1 - \frac{1}{1+r} \;\Longrightarrow\; \tfrac{1}{2}\Big[\ln(1+r) + \frac{1}{1+r} - 1\Big] > 0, \qquad r = \frac{a^2}{\sigma^2} > 0
```

So the interventional gain separates A from B, while the observational gain cannot.

### P5. Generality bound

If an agent scores at most epsilon on every environment outside a set S, its total score is capped by the prior mass of S.

```math
\Upsilon^{\mathrm{N}}(A) \le \sum_{\mu \in S} \xi(\mu) \cdot 1 + \sum_{\mu \notin S} \xi(\mu) \cdot \varepsilon \le \xi(S) + \varepsilon
```

This uses gain at most 1 (P1) and total prior mass at most 1 (Kraft's inequality, for a prefix-free DSL). The JPEG-style specialist is therefore capped by its class's share of the prior.

## MVP experiment

Eight agents on three environment families test H1 and H3 in phase 1; the design is small enough to fit on one GPU, and H4 waits for the causal family in phase 2.

**Environment families.** Each environment is drawn by sampling a program from a probabilistic grammar over the DSL, so it appears with probability 2 to the power of minus its description length. A second DSL with different primitives is built for H3.

| Family | Generator | Description length grows with | Role |
| --- | --- | --- | --- |
| E1 Markov chains | Order 0–4, binary alphabet, quantized transition probabilities | Number of parameters (2 to the power k) | Statistical structure |
| E2 Rule sequences | Periodic patterns; recurrences such as x(t) = x(t−a) XOR x(t−b) | Pattern length, rule size | Discrete, program-like structure (ARC-like) |
| E3 Noisy low-dimensional dynamics | Discretized linear dynamical system, 1–3 latent dimensions | Latent dimension, precision | Your manifold case |
| E4 Causal models (phase 2) | Linear-Gaussian SCMs, 3–5 variables, held-out interventions | Graph size | Interventional gain |

**Agents.** Two anchors calibrate the scale; six contestants map onto the test cases.

| Agent | What it is | Why it is in the set |
| --- | --- | --- |
| Uniform | Guesses uniformly | Anchor at 0 |
| Bayes oracle | Knows the true environment | Anchor at 1, not a contestant |
| Memorizer | Unsmoothed n-gram, exact-match lookup | Lookup-table case |
| gzip predictor | LZ-based compressor used as a predictor | General but shallow |
| CTW | Context-tree weighting, depth 8 | Principled near-optimal baseline for E1 |
| Specialist | Hard-coded for periodic E2 patterns only | JPEG case |
| Meta-trained transformer | Small model trained across E1–E3, predicts in context | Modern learner |
| Bounded program search | Enumerates DSL programs up to a length cap, Bayesian mixture | Resource-bounded Solomonoff |

**Proposed defaults** (adjust after a pilot): 1,000 environments per family, n from 1 to 1,024 in doublings, a held-out block of m = 64, 5 seeds. Cost = code size plus parameter count times bits per parameter, plus log2 of FLOPs.

**Hypotheses** (preregister before running):

1. **H1 Case verdicts.** Ranking: program search ≈ CTW > transformer > gzip > specialist > memorizer ≈ uniform. The specialist scores high only on E2.
2. **H2 Agreement (optional, phase 1.5).** With log-score rewards, a Legg–Hutter proxy agrees with the proposed measure (Kendall tau at least 0.8); adding a bandit family where reward is not prediction lowers tau.
3. **H3 Reference robustness.** Kendall tau between rankings under DSL-1 and DSL-2 is at least 0.8.
4. **H4 Causal (phase 2).** On Markov-equivalent SCM pairs, the observational gain gap between causal and correlational agents is below 0.01 while the interventional gap exceeds 0.1.

**Kill and pivot criteria.**

- **Anchors do not land near 0 and 1:** pipeline bug. Stop and fix.
- **Memorizer or gzip outranks CTW and program search:** the normalization or learning-curve weights are wrong. Revisit Steps 3–4 before anything else.
- **H3 fails badly (tau below 0.5):** the measure is too language-dependent at this scale. Publish that as a negative result on reference dependence in practice, and report only rankings that hold across both DSLs.

## Fit, limitations and plan

This project is the definitional twin of the ARC compression project and should stay a separate paper.

### Relation to the ARC project

- **Different questions.** The ARC project asks an empirical question: does causally relevant compression predict out-of-distribution generalization? This project asks what a definition of intelligence built on that idea should be, and what provably follows from it.
- **Shared machinery.** Interventional tests (Step 7 here, the do(C) and do(U) measurements there), MDL baselines and planted generator suites can be reused across both.
- **Keep the firewall.** The ARC plan deliberately keeps intelligence-as-compression out of its published claims. This project makes that claim the object of study, so a separate paper lets the ARC results stand on their own.

### Limitations

- **Prediction over action.** P3 shows the measure rates the Oracle highly. That is a philosophical commitment to an epistemic view of intelligence, to defend in the paper rather than hide.
- **No single number across budgets.** Budget classes replace an arbitrary trade-off constant, at the cost of a frontier instead of a scalar.
- **The DSL prior only approximates Kolmogorov complexity.** H3 measures how much that matters.
- **Alpha is a free parameter** once the interventional term is added.
- **Sequences only.** Open-ended, multi-agent and embodied environments are out of scope.

### Timeline

Phases 0–3 take six weeks, each ending at a gate; the causal family and the bandit comparison come after.

The H3 gate is the real decision point: if rankings depend on the reference language, the project pivots to a negative result.

### Venues

- **Conceptual paper** (sections 3, 5, 6, 7): Minds and Machines, or the journal Philosophy of AI.
- **Formal measure plus benchmark** (sections 4, 7, 8): the AGI conference, or a NeurIPS or ICLR workshop on evaluation or foundations.

### References

- Carnap, R. (1950). *Logical Foundations of Probability.* (Explication as method.)
- Chollet, F. (2019). On the Measure of Intelligence.
- Delétang, G. et al. (2023). Language Modeling Is Compression.
- Grünwald, P. (2007). *The Minimum Description Length Principle.*
- Hutter, M. (2005). *Universal Artificial Intelligence.*
- Legg, S. and Hutter, M. (2007). Universal Intelligence: A Definition of Machine Intelligence. *Minds and Machines.*
- Leike, J. and Hutter, M. (2015). Bad Universal Priors and Notions of Optimality. *COLT.*
- Pearl, J. (2009). *Causality*, 2nd ed.
- Solomonoff, R. (1964). A Formal Theory of Inductive Inference.
- Willems, F., Shtarkov, Y. and Tjalkens, T. (1995). The Context-Tree Weighting Method.
