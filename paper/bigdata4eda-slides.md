---
title: "Big Data for EDA 🧮"
subtitle: "Probabilistic Sketches & GPU Acceleration for Chip Analysis ⚡🔋"
author: "Wai-Shing Luk 👨‍💻"
date: "2026 📅"
---

## 🗺️ Agenda

1. 🔥 Why approximate? The EDA data explosion
2. 🪣 **Count-Min Sketch** — frequency estimation
3. 🎯 **HyperLogLog** — cardinality estimation
4. ⚡ **GPU acceleration** with Numba CUDA
5. 🔋 **Case study**: switching power analysis
6. 📊 **Results**: estimates vs. ground truth
7. 🤖 Big data for **AI-era** chip design
8. 🚀 Lessons & next steps

# 🔥 Part 1 — Why Approximate?

## 📈 The EDA data explosion

- Modern SoCs: **1M+ gates**, **500k nets**, **10 layers** 🔩
- Logic simulation: **$10^9$ cycles** $\times$ millions of nets 🔁
- Every node toggle is a streaming event $\to$ exact bookkeeping is **$O(n)$** 💥

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=7mm and 8mm]
  \node[ngreen] (rtl) {RTL Netlist\\1M+ gates};
  \node[nblue, right=of rtl] (sim) {Simulation\\$10^9$ cycles};
  \node[nyellow, right=of sim] (ev) {Toggle-Event Stream\\100M+ events};
  \node[npurple, right=of ev] (sk) {Probabilistic Sketches\\$O(1)$ memory};
  \node[nred, right=of sk, yshift=7mm] (hot) {Hot-spot Map};
  \node[nred, right=of sk, yshift=-7mm] (pwr) {Power Estimate};
  \draw[ar] (rtl) -- (sim);
  \draw[ar] (sim) -- (ev);
  \draw[ar] (ev) -- (sk);
  \draw[ar] (sk) -- (hot);
  \draw[ar] (sk) -- (pwr);
\end{tikzpicture}}
\end{center}
```

## ⚖️ Exact vs. approximate

| | Exact counting 📏 | Sketch 🪣 |
|:--|:--|:--|
| Memory | $O(n)$ | $O\!\big(\tfrac{1}{\varepsilon}\log\tfrac{1}{\delta}\big)$ |
| Query | exact | probabilistic bound |
| One-pass stream | $\times$ needs storage | $\checkmark$ |

Key idea: **trade a tiny, bounded error for a huge memory saving** 🎯

$$ \Pr\big[\,\hat{c}_x \le c_x + \varepsilon N\,\big] \;\ge\; 1-\delta $$

where $N$ is the total number of events. 🔬

# 🪣 Part 2 — Count-Min Sketch

## 🪣 Count-Min Sketch: intuition

Use a **small 2-D array** of counters and $d$ independent hash functions.
Increment on every event; query returns the **minimum** across rows.

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=6mm and 8mm]
  \node[ngreen] (x) {Event $x$};
  \node[nblue, right=of x] (h) {$d$ independent hashes\\$h_1 \dots h_d$};
  \node[nyellow, right=of h] (c) {Increment one counter\\per row};
  \node[ngreen, below=8mm of x] (y) {Query $y$};
  \node[nblue, right=of y] (h2) {same $d$ hashes};
  \node[nred, right=of h2] (m) {$\min$ over $d$ rows\\never underestimates $\checkmark$};
  \draw[ar] (x) -- (h);
  \draw[ar] (h) -- (c);
  \draw[ar] (y) -- (h2);
  \draw[ar] (h2) -- (m);
\end{tikzpicture}}
\end{center}
```

Collisions only ever **over-count**, so the minimum is a safe estimate. 🛟

## 📐 Count-Min Sketch: algorithm

**Universal hash** (Mersenne prime $p = 2^{31}-1$):

$$ h_i(x) = \big((a_i x + b_i) \bmod p\big) \bmod w $$

**Update** (additive, all rows):

$$ \forall i \in [d]:\quad C[i,\, h_i(x)] \mathrel{+}= 1 $$

**Query** (minimum = over-estimate):

$$ \hat{c}_x = \min_{i \in [d]}\ C[i,\, h_i(x)] $$

**Guarantee:** with probability $\ge 1-\delta$, $\ {\color{nordblue}\hat{c}_x \le c_x + \varepsilon N}$

## 📏 Sizing the sketch

Choose $w = \lceil e/\varepsilon \rceil$ and $d = \lceil \ln(1/\delta)\rceil$.

For $\varepsilon = 0.001$ and $\delta = 0.01$:

$$ w = \left\lceil \frac{e}{0.001} \right\rceil = {\color{nordgreen}2719},\qquad d = \big\lceil \ln 100 \big\rceil = {\color{nordgreen}5} $$

$$ \text{memory} = w \cdot d \cdot 4\,\text{B} = 2719 \cdot 5 \cdot 4 = 54{,}380\,\text{B} \;\approx\; {\color{nordyellow}53.1\,\text{KB}} $$

Additive error budget $\varepsilon N$ — e.g. $N=10^8 \Rightarrow \pm 10^5$. 📉

## 🔬 Demo: switching-activity tracking

CM sketch with only $w=50,\ d=3$ tracks 6 nets over **235 toggle events**:

| Net | Node ID | Toggles | Estimate |
|:--|--:|--:|--:|
| `clk_buffer_1` | 1001 | 150 | ✅ 150.0 |
| `data_reg_1` | 2001 | 75 | ✅ 75.0 |
| `memory_cell` | 4001 | 10 | ✅ 10.0 |
| others | — | 0 | 0.0 |

Recovers exact counts **with no collisions** — hot spots are obvious after one pass. 🌡️

# 🎯 Part 3 — HyperLogLog

## 🎯 HyperLogLog: intuition

Idea: if a uniform hash has $k$ **leading zeros**, you have probably seen about $2^{k}$ distinct items. 👀

Track the **maximum** rank per register; combine registers with a **harmonic mean**.

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=6mm and 7mm]
  \node[ngreen] (x) {Item};
  \node[nblue, right=of x] (hash) {64-bit hash};
  \node[npurple, right=of hash, yshift=8mm] (idx) {top $p$ bits\\register $j$};
  \node[nyellow, right=of hash, yshift=-8mm] (rem) {remaining bits\\rank $\rho$};
  \node[npurple, right=of idx, yshift=-8mm] (upd) {$M[j]=\max(M[j],\rho)$};
  \node[nyellow, right=of upd] (est) {$\hat{E}=\alpha_m m^2/\sum 2^{-M[j]}$};
  \draw[ar] (x) -- (hash);
  \draw[ar] (hash) -- (idx);
  \draw[ar] (hash) -- (rem);
  \draw[ar] (idx) -- (upd);
  \draw[ar] (rem) -- (upd);
  \draw[ar] (upd) -- (est);
\end{tikzpicture}}
\end{center}
```

Memory is just $m = 2^p$ small registers — **kilobytes** for millions of items. 💾

## 📐 HyperLogLog: the math

Registers $m = 2^{p}$ (precision $p \in [4,16]$, default $14 \Rightarrow m = 16{,}384$).

$$ j = \text{top } p \text{ bits of } h(x),\qquad \rho(x) = \#\text{leading zeros of remaining bits} $$

$$ M[j] = \max\big(M[j],\, \rho(x)\big) $$

**Estimator** (harmonic mean), with $\alpha_m = \dfrac{0.7213}{1 + 1.079/m}$:

$$ \hat{E} = {\color{nordblue}\alpha_m \cdot \frac{m^{2}}{\sum_{j=1}^{m} 2^{-M[j]}}} $$

## 🔧 Bias corrections & error

**Small range** (linear counting), when $\hat{E} \le \tfrac{5}{2} m$:

$$ {\color{nordgreen}\hat{E} = m \ln\!\left(\frac{m}{V}\right)},\qquad V = \#\{\text{zero registers}\} $$

**Large range**, when $\hat{E} > \tfrac{1}{30}2^{64}$:

$$ {\color{nordred}\hat{E} = -2^{64}\ln\!\left(1 - \hat{E}/2^{64}\right)} $$

Standard error is essentially independent of the cardinality:

$$ \sigma \approx \frac{1.04}{\sqrt{m}} \;\Rightarrow\; 2.6\% \text{ at } p{=}10,\ \ 0.8\% \text{ at } p{=}14 $$

## 💾 Memory vs. accuracy: theoretical

$$ \sigma \approx \frac{1.04}{\sqrt{m}},\qquad \text{memory} = 4\,m\ \text{bytes (int32 registers)} $$

| $p$ | $m$ | Memory | Std. error |
|--:|--:|--:|--:|
| 8 | 256 | 1 KB | 6.50% |
| 10 | 1,024 | 4 KB | 3.25% |
| 12 | 4,096 | 16 KB | 1.63% |
| 14 | 16,384 | 64 KB | 0.81% |
| 16 | 65,536 | 256 KB | 0.41% |

**256 KB estimates cardinalities of essentially any size.** 🌍

## 📊 Demo: measured accuracy

Measured on CPU (SHA-256 hashing), distinct-items benchmark:

| $p$ | $m$ | Memory | Measured error |
|--:|--:|--:|--:|
| 8 | 256 | 0.2 KB | 0.98% |
| 10 | 1,024 | 0.8 KB | 2.20% |
| 12 | 4,096 | 3.0 KB | 2.73% |
| 14 | 16,384 | 12 KB | 0.46% |
| 16 | 65,536 | 48 KB | 0.02% |

- Merge of two disjoint 5,000-sets $\to$ est. 10,457 vs. true 10,000 (**4.6% error**) 🔗
- 1M random draws $\to$ est. 99,470 distinct in **~3 KB** 🎉

## 🏗️ HyperLogLog in EDA: application suite

One cardinality primitive, **seven** EDA estimators (map objects $\to$ typed signatures, then count unique):

| Application | Estimates |
|:--|:--|
| 🔎 DRC | unique violation patterns |
| 🕸️ Netlist connectivity | unique signal paths $\cdot$ fanout $\cdot$ critical paths |
| 🗺️ Placement | unique cell types per region |
| ⏱️ Timing | unique paths $\cdot$ critical paths |
| 🔋 Power activity | unique activity patterns $\cdot$ transition counts |
| ✅ Verification | unique states reached $\cdot$ coverage % |
| 🏭 Yield | unique defect patterns $\cdot$ failing dies |

All with just $m = 2^{p}$ registers. Coverage, congestion and yield pivot on the same math. 🎯

# ⚡ Part 4 — GPU Acceleration

## 🚀 Why GPU?

- Hashing + register updates are **embarrassingly parallel** 🧵
- A CPU handles one element at a time; a GPU runs **millions of threads** at once
- Numba `@cuda.jit` compiles Python kernels to native CUDA — **no PyTorch needed** 🐍

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=7mm and 7mm]
  \node[ngreen] (d) {Batch of hashes\\up to 1M};
  \node[nblue, right=of d] (grid) {Grid: $\lceil n/256\rceil$ blocks};
  \node[nyellow, right=of grid] (t) {256 threads / block};
  \node[nred, right=of t] (a) {atomicMax(registers[$j$], $\rho$)};
  \node[npurple, below=8mm of grid] (r) {Registers in global memory};
  \node[nblue, right=of r] (c) {copy\_to\_host()};
  \node[npurple, right=of c] (e) {Harmonic-mean estimate};
  \draw[ar] (d) -- (grid);
  \draw[ar] (grid) -- (t);
  \draw[ar] (t) -- (a);
  \draw[ar] (a.south) |- (r.east);
  \draw[ar] (r) -- (c);
  \draw[ar] (c) -- (e);
\end{tikzpicture}}
\end{center}
```

## 🛠️ GPU HyperLogLog: kernel design

- One thread per hash; `cuda.grid(1)` gives the thread id 🧮
- Register index from top $p$ bits, rank from the rest

$$ j = h(x) \gg (64-p),\qquad \rho = \text{leading-zeros of } h(x) \ll p $$

- **Race-free** update via hardware atomics:

$$ {\color{nordblue}M[j] \leftarrow \texttt{atomicMax}\big(M[j],\, \rho\big)} $$

- Hashes precomputed on CPU with **MurmurHash3 x64**, uploaded via `cuda.to_device` 📤

## 🔡 Hashing on the GPU

Two hash families in the demos:

- **xorshift** (`simple_gpu_demo`): `x ^= x<<13; x ^= x>>7; x ^= x<<17` — fast, low quality ⚡
- **MurmurHash3 fmix64** (`better_gpu_demo`, `hyper_loglog_gpu_demo`) — well-mixed 🏆

$$ k \leftarrow k \oplus (k \gg 33);\quad k \leftarrow k \cdot \texttt{0xFF51AFD7ED558CCD};\ \dots $$

| Kernel | Buckets | Target error |
|:--|--:|--:|
| `b = 14` | 16,384 | $\approx$ 1% |
| `b = 16` | 65,536 | $\approx$ 0.4% |

## ⚡ GPU demos

- `simple_gpu_demo`: **10M** items, $b=14$ (16,384 buckets) 🧮
- `better_gpu_demo`: **10M** items, with linear-counting correction 🎯
- `hyper_loglog_gpu_demo` benchmark: **5M** items, 500k unique, batch sizes 1k $\to$ 1M 📦
- Memory always $m \times 4$ bytes — 64 KB holds 16,384 registers 💾

> Throughput / speed-up depend on the GPU; the demo reports `cpu_time / gpu_time` at runtime. ⏱️

# 🔋 Part 5 — Switching Power Case Study

## 🔋 From toggles to power

Activity factor from the sketch, then the classic dynamic-power formula:

$$ \alpha = \frac{n_{\text{toggles}}}{N_{\text{cycles}}},\qquad {\color{nordred}P = \alpha\, C\, V^{2}\, f} $$

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=7mm and 8mm]
  \node[ngreen] (n) {Netlist + node caps};
  \node[ngreen, right=of n] (t) {Toggle stream};
  \node[npurple, right=of t] (cms) {Count-Min Sketch};
  \node[nblue, right=of cms] (a) {$\alpha=$ count / cycles};
  \node[nred, right=of a] (p) {$P = \alpha C V^2 f$};
  \node[nyellow, right=of p] (r) {Hot-spot + power report};
  \draw[ar] (n) -- (t);
  \draw[ar] (t) -- (cms);
  \draw[ar] (cms) -- (a);
  \draw[ar] (a) -- (p);
  \draw[ar] (p) -- (r);
\end{tikzpicture}}
\end{center}
```

## 🏭 Demo: full-chip power analysis

- **500 nodes**, 1,000 cycles, **64,106 toggle events** in **0.69 s** ⏱️
- CMS auto-sized to $w=2719,\ d=5$ (**53.1 KB**) from $\varepsilon=0.001,\ \delta=0.01$

| Category | Nodes | $C$ | Total power |
|:--|--:|--:|--:|
| 🕐 clock | 50 | 5 fF | **125.0 $\mu$W** |
| 🧠 logic | 150 | 2 fF | 60.5 $\mu$W |
| 🎛️ control | 100 | 1 fF | 2.9 $\mu$W |
| 💾 memory | 200 | 1 fF | 5.9 $\mu$W |

Hot nodes = the clock tree ($\alpha = 0.5$, $2.5\,\mu\text{W}$ each) 🌡️

## 📊 Generated: switching power analysis

```{=latex}
\begin{center}
\includegraphics[width=0.72\textwidth]{figures/switching_power_analysis.png}
\end{center}
```

Power by category and the top power-hungry nodes — from a 53.1 KB sketch. 🌡️

## ⏳ Temporal activity

Windows of 100 cycles, per-window sketches ($w=500,\ d=4$). Two signatures:

**Burstiness** — spikes vs. steady traffic:

$$ {\color{nordyellow}B = \frac{\sigma - \mu}{\sigma + \mu}} $$

**Periodicity** — coefficient of variation of inter-event intervals:

$$ c_v = \frac{\sigma_{\text{interval}}}{\mu_{\text{interval}}},\qquad \text{periodic if } c_v < 0.3 $$

| Node | Burstiness | Mean $\alpha$ |
|:--|--:|--:|
| periodic_clock | 0.00 | 0.50 |
| bursty_node | 0.35 | 0.16 |
| random_node | 0.30 | 0.17 |
| quiet_node | 0.50 | 0.02 |

## ⚡ Current envelope

Peak switching current from charge/discharge:

$$ {\color{nordred}I_{\text{peak}} = \frac{C\,V}{\Delta t}},\qquad \Delta t = \text{transition time} = 50\ \text{ps} $$

- Envelope = sum of triangular pulses over hot nodes 🔺
- Hot-node filter: activity $\alpha > 0.1$, top-20 nodes 🔝
- Demo: **peak 0.34 mA**, average 0.04 mA 📈

Sanity-check IR-drop / current-density risks early. 🛡️

## 📊 Generated: temporal & current profile

```{=latex}
\begin{center}
\includegraphics[width=0.54\textwidth]{figures/temporal_current_analysis.png}
\end{center}
```

Per-node activity over time and the estimated current envelope. ⚡

## 🎼 Pattern-dependent analysis

Different workloads stress different nodes. **Sensitivity index:**

$$ {\color{nordgreen}\text{SI} = \frac{\max_k a_k - \min_k a_k}{\text{mean}_k a_k}} $$

- Patterns: `Idle`, `Reset`, `Max_Load`, `Worst_Case` 🎭
- Toggles: Idle 1,286 $\cdot$ Reset 2,250 $\cdot$ Max_Load 3,121 $\cdot$ **Worst_Case 3,997**
- Most sensitive nodes: $\text{SI} = 4.0$, worst pattern = **Max_Load**
- Correlation matrix groups co-active nodes 🔗

## 🧊🔥 Multi-corner (PVT) analysis

Same design across process / voltage / temperature corners:

| Corner | $V$ | $T$ ($^\circ$C) | Process |
|:--|--:|--:|:--|
| `TT_25C` | 1.0 | 25 | typical |
| `FF_125C` | 1.1 | 125 | fast |
| `SS_-40C` | 0.9 | -40 | slow |
| `Turbo_Mode` | 1.0 | — | boost |

Sample node (data path): TT 53.0 $\mu$W $\cdot$ FF **72.6 $\mu$W** $\cdot$ SS 28.5 $\mu$W $\cdot$ Turbo 49.5 $\mu$W

$$ {\color{nordred}\text{power variation} = 86.6\%},\qquad \text{worst corner} = \texttt{FF\_125C} $$

## 📊 Generated: pattern & corner analysis

```{=latex}
\begin{center}
\includegraphics[width=0.54\textwidth]{figures/pattern_corner_analysis.png}
\end{center}
```

Pattern sensitivity, multi-corner power and co-activity correlation. 🎭

## ⚖️ Memory $\leftrightarrow$ accuracy trade-off

10,000-node power-law workload, **119,266 events**:

| Config | Counters | Memory | Avg error | Hot-node error |
|:--|--:|--:|--:|--:|
| Tiny | 300 | 1.2 KB | 26,538% | 323% |
| Small | 2,000 | 7.8 KB | 3,380% | 41% |
| Medium | 10,000 | 39 KB | 349% | **5.8%** |
| Large | 60,000 | 234 KB | 0.16% | **0.01%** |

**Punch-line:** hot-spot *ranking* is robust even at medium size ✅ — but over-shrinking destroys the tails. 🚨

## 📊 Generated: memory $\leftrightarrow$ accuracy

```{=latex}
\begin{center}
\includegraphics[width=0.54\textwidth]{figures/memory_accuracy_tradeoff.png}
\end{center}
```

Sketch size versus estimation error — the trade-off made precise. ⚖️

# 📊 Part 6 — Results

## 📐 EDA application suite: estimates vs. ground truth

```{=latex}
\scriptsize
```

Twelve metrics, each from a single-pass HyperLogLog sketch (v0.2.0 run):

| Metric | Estimate | Actual | Error |
|:--|--:|--:|--:|
| 🔎 DRC patterns | 300,877 | 290,919 | 3.4% |
| 🕸️ Signal paths | 498,998 | 500,000 | 0.2% |
| 🔀 Fanout degrees | 49 | 49 | 0.0% |
| 🗺️ Cell types / region | 602 | 600 | 0.3% |
| ⏱️ Timing paths | 99,503 | 100,000 | 0.5% |
| ⚠️ Critical paths | 49,856 | 50,584 | 1.4% |
| 🔋 Activity patterns | 9,888 | 9,964 | 0.8% |
| 📈 Transition counts | 63 | 63 | 0.0% |
| ✅ States covered | 256 | 256 | 0.0% |
| 🔁 Transitions covered | 65,831 | 65,506 | 0.5% |
| 🏭 Defect patterns | 138 | 140 | 1.4% |
| 💀 Failing-die buckets | 636 | 637 | 0.2% |

All errors stay within HyperLogLog's $\sigma \approx 1.04/\sqrt{m}$ bound. 🎯

## 🏁 Results at a glance

- 🪣 **Count-Min Sketch**: recovers 235 toggle events exactly at $w{=}50,\ d{=}3$
- 🏭 **Full-chip power**: 500 nodes, **64,106 events $\to$ 53.1 KB sketch in 0.59 s**; clock tree = 125 $\mu$W of 194 $\mu$W (64%) 🌡️
- ⏳ **Temporal**: bursty node $B = 0.35$; periodic clock detected (confidence 1.00)
- ⚡ **Current envelope**: peak **0.34 mA**, average 0.04 mA
- 🧊🔥 **PVT corners**: **86.6%** power variation, worst corner `FF_125C`
- ⚖️ **Memory $\leftrightarrow$ accuracy**: 39 KB sketch $\to$ **4–6%** hot-node error; 234 KB $\to$ **0.01%**
- 🎯 **HyperLogLog**: 12 KB $\to$ **<0.5%** error ($p{=}14$); **1M items $\to$ 3 KB**
- 📐 **EDA suite**: 12 metrics, all $\le$ 3.4% error, most $<$ 1% ✅

Zero stored elements — every number is computed on the fly. 🪄

# 🤖 Part 7 — Big Data for AI-Era Chip Design

## 🤖 Chip design in the AI era

- AI workloads demand **enormous silicon**: 100B+ transistors, chiplets, 2 nm 🔩
- **AI-generated RTL** and ML-driven design-space exploration multiply data volume faster than tools scale 📈
- Verification state spaces are **astronomical** — coverage closure dominates schedules 🕳️
- Design decisions must now be made **in real time**, inside automated / AI loops ⏱️

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=6mm and 6mm]
  \node[npurple] (ai) {AI workloads\\LLMs, training};
  \node[nblue, right=of ai] (chip) {Huge AI chips\\100B+ transistors};
  \node[nyellow, right=of chip] (data) {Design data\\sim $\cdot$ DRC $\cdot$ timing};
  \node[ngreen, right=of data] (an) {Streaming analytics\\sketches + GPU};
  \node[nred, right=of an] (dec) {Real-time\\DSE \& sign-off};
  \draw[ar] (ai) -- (chip);
  \draw[ar] (chip) -- (data);
  \draw[ar] (data) -- (an);
  \draw[ar] (an) -- (dec);
  \draw[ar] (dec.south) -- ++(0,-5mm) -| (ai.south);
\end{tikzpicture}}
\end{center}
```

## 🏗️ Why sketches + GPU are the enabling layer

- 🌊 **Streaming, single pass** — fit simulation, DRC and toggle-event streams without storage
- 🪶 **$O(1)$ memory** — analyze billion-gate designs on a workstation, not a cluster
- 🔗 **Mergeable** — tile across chiplets, cores and cloud for distributed analysis
- ⚡ **GPU-accelerable** — thousands of parallel hashes/updates for AI-scale data
- 🎲 **Bounded error** — statistical guarantees ($\sigma \approx 1.04/\sqrt{m}$) for automated decisions
- 🧱 **Foundation for AI-EDA** — fast, scalable metrics feed ML surrogate models and RL placers/routers

Sketches turn **unbounded EDA data into bounded, decision-ready signals** for AI-driven flows. 🤝

## 🔁 From sketches to AI-driven EDA

```{=latex}
\begin{center}
\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%
\begin{tikzpicture}[node distance=6mm and 6mm]
  \node[nyellow] (raw) {EDA data streams\\netlist $\cdot$ sim $\cdot$ DRC};
  \node[npurple, right=of raw] (sk) {Sketches\\CMS $\cdot$ HLL};
  \node[ngreen, right=of sk] (feat) {Scalable metrics\\hot spots $\cdot$ coverage};
  \node[nblue, right=of feat] (ml) {ML / AI models\\surrogate $\cdot$ RL $\cdot$ LLM};
  \node[nred, right=of ml] (act) {Design actions\\place $\cdot$ route $\cdot$ opt.};
  \draw[ar] (raw) -- (sk);
  \draw[ar] (sk) -- (feat);
  \draw[ar] (feat) -- (ml);
  \draw[ar] (ml) -- (act);
  \draw[ar] (act.south) -- ++(0,-5mm) -| (raw.south);
\end{tikzpicture}}
\end{center}
```

- Sketches are the **memory-bounded sensing layer** that makes the AI feedback loop affordable 🪄
- The same math drives **verification closure, congestion, yield and power** — one toolkit, many decisions 🧰

## 💡 Lessons learned

- ✅ One pass, bounded memory — sketches fit streaming EDA data
- ✅ Count-Min Sketch $\to$ frequencies & hot spots; HyperLogLog $\to$ unique counts
- ✅ GPU atomics make register updates race-free and fast ⚛️
- ⚠️ Size accuracy to the *query*: hot-spot ranking is robust, tails are not
- ⚠️ Reproducibility: seed both NumPy **and** the sketch hash RNG 🎲
- 🚧 Watch for subtle bugs (e.g. merge re-hashing, corner defaults)

> Pick the sketch size from the error you can *tolerate*. 🎯

## 🎤 Q&A

### Thanks for listening! 🙏

**Wai-Shing Luk** 👨‍💻 · 2026 📅
