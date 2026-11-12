// Builds report/FairShed_Report.docx from results/results.json and figures/*.png.
// Usage:  npm install docx  &&  node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, HeadingLevel, LevelFormat, BorderStyle, ExternalHyperlink,
  Footer, PageNumber,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "results.json"), "utf8"));
const FONT = "Calibri";
const CONTENT_W = 9026; // A4 with 1-inch margins, in DXA
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const f1 = (x) => Number(x).toFixed(1);

// ---------- helpers ----------
const p = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 276 },
    ...opts.para,
    children: (Array.isArray(text) ? text : [text]).map((t) =>
      typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 22, ...opts.run }) : t),
  });
const b = (text) => new TextRun({ text, bold: true, font: FONT, size: 22 });
const i = (text) => new TextRun({ text, italics: true, font: FONT, size: 22 });
const mono = (text) => new TextRun({ text, font: "Consolas", size: 20 });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const bullet = (parts) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 276 },
    children: (Array.isArray(parts) ? parts : [parts]).map((t) =>
      typeof t === "string" ? new TextRun({ text: t, font: FONT, size: 22 }) : t),
  });
const formula = (text) =>
  new Paragraph({
    spacing: { before: 60, after: 60 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Cambria Math", size: 22 })],
  });
const link = (text, url) =>
  new ExternalHyperlink({ link: url, children: [new TextRun({ text, style: "Hyperlink", font: FONT, size: 22 })] });

function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), buf };
}
function figure(name, caption, widthPx = 600) {
  const { w, h, buf } = pngSize(path.join(ROOT, "figures", name));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round((widthPx * h) / w) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, font: FONT, size: 19, color: "52514E" })],
    }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "C3C2B7" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, c) => a + c, 0);
  const cell = (text, idx, head) =>
    new TableCell({
      borders,
      width: { size: widths[idx], type: WidthType.DXA },
      shading: head ? { fill: "E8EEF7", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({
        alignment: idx === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [new TextRun({ text: String(text), bold: head, font: FONT, size: 20 })],
      })],
    });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((t, k) => cell(t, k, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((t, k) => cell(t, k, false)) })),
    ],
  });
}
const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------- numbers ----------
const inst = R.instance, W = inst.weights;
const con = R.qaoa.filter((r) => r.mode === "constrained");
const pen = R.qaoa.filter((r) => r.mode === "penalty");
const cLast = con[con.length - 1], pLast = pen[pen.length - 1];
const uniValid = R.uniform_valid_p_optimal;
const big = R.larger_instance;
const mw = 5;
const sweepRows = R.tradeoff.map((s) => [
  s.fairness_scale + "×", f1(s.exact_social), s.exact_outages.join(" / "),
  s.exact_consecutive, s.qaoa_matches_exact ? "Yes" : "No",
]);

// ---------- document ----------
const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun({ text: "FairShed", font: FONT })] }),
  p("A quantum (QAOA) scheduler for fair load shedding", { run: { size: 28, color: "52514E" }, para: { spacing: { after: 60 } } }),
  p("Qiskit Fall Fest 2026 Hackathon · Domain 6: Energy & Smart Infrastructure", { run: { color: "52514E" }, para: { spacing: { after: 60 } } }),
  p([b("Team: "), "Runtime Terrors: Abdul Zakir (team lead), Muhammad Kashif, Huzefa Aslam"], { para: { spacing: { after: 40 } } }),
  p([b("Code: "), link("github.com/abzakir/FairShed", "https://github.com/abzakir/FairShed")], { para: { spacing: { after: 40 } } }),
  p([b("Live app: "), link("fairshed.streamlit.app", "https://fairshed.streamlit.app")], { para: { spacing: { after: 40 } } }),
  p([b("Video (3 min): "), link("Google Drive", "https://drive.google.com/file/d/1GNZddvKO6-_PV2KvJzHGPbF4FQYLX6p4/view?usp=sharing")], { para: { spacing: { after: 240 } } }),

  h1("Summary"),
  p("When a distribution company in Pakistan runs short of power, someone has to decide which feeders go dark in each hour. FairShed turns that decision into a small optimisation problem with two goals that pull against each other: keep the harm low, and share the outages fairly. We solve it with QAOA in Qiskit and check every answer against an exact solver."),
  p(`On our test case (${inst.feeders.length} feeder blocks over ${inst.slots.length} evening hours, ${inst.qubits} qubits, ${inst.schedules_total.toLocaleString()} possible schedules) the best schedule we measured from the quantum circuit ${cLast.sampled_best_is_optimal ? "is the exact optimum" : "is close to the exact optimum"}. The result we care most about is how we built the circuit. Textbook QAOA puts the rule "cover the shortfall exactly" into the cost function as a penalty. We built that rule into the circuit instead. At ${cLast.p} QAOA layers this raised the chance that a single measurement is the optimum from ${pct(pLast.p_optimal, 2)} to ${pct(cLast.p_optimal, 1)}, and every noise-free measurement obeyed the rule.`),
  p("At this size a laptop finds the optimum by brute force in milliseconds, so we make no claim of a quantum speedup. What we can show is a correct formulation, a working implementation and app, and measurements of where the method holds up and where it breaks."),

  h1("1. Problem and who it is for"),
  p("Load shedding in Pakistan is planned feeder by feeder, and the hours of outage on a feeder are tied to its revenue losses from theft and unpaid bills. NEPRA, the regulator, reviewed the distribution companies for FY2024-25 and called these revenue-loss-based cuts illegal, finding that they had not delivered results [1]. A fixed priority list also fails in a simple way: the same neighbourhoods lose power hour after hour while others are never touched."),
  p("Our user is a control-room operator at a distribution company. They receive a forecast of the shortfall for each of the next few hours and have to publish a shedding schedule."),
  p("The schedule has to cover the shortfall in every hour and keep critical loads such as hospitals running. Beyond that, two goals conflict. The cheapest schedule keeps cutting the same low-priority feeder. A fair schedule spreads the outages, which sometimes means cutting a feeder whose outage costs more. FairShed lets the operator decide how much weight fairness gets."),

  h1("2. Optimisation formulation"),
  h2("Decision variables"),
  p([mono("x[f,t] = 1"), " if feeder block f is switched off in time slot t, otherwise 0. With F feeders and T slots there are F × T binary variables, one qubit each."]),
  h2("Objective (minimise)"),
  formula("E(x) = Σ c_f · x[f,t]                          social cost of the outages"),
  formula("     + B · Σ x[f,t] · x[f,t+1]                  back-to-back outages on one feeder"),
  formula("     + C · Σ_f ( h_f + Σ_t x[f,t] )²            outages concentrated on one feeder"),
  formula("     + A · Σ_t ( Σ_f P_f · x[f,t] − D_t )²      shortfall not covered exactly"),
  p("Here c_f is the harm caused by one slot of outage on feeder f, P_f is its load, D_t is the shortfall in slot t, and h_f is the number of slots feeder f was already off earlier that day (0 in our test case). The squared term is smallest when outages are spread evenly, which is what makes it a fairness penalty. Every term is at most quadratic, so the problem is a QUBO and needs no slack variables."),
  h2("Constraint"),
  p("In every slot the load switched off must equal the shortfall: Σ_f P_f · x[f,t] = D_t. The penalty version enforces this through the A term. Our main version enforces it inside the circuit (Section 3)."),
  h2("Test instance"),
  p(`${inst.feeders.length} feeder blocks of ${mw} MW each over ${inst.slots.length} evening-peak slots, so ${inst.qubits} variables. Shortfall: ${inst.deficits.map((d, k) => `${d * mw} MW at ${inst.slots[k]}`).join(", ")}. Weights: A = ${W.deficit}, B = ${W.consecutive}, C = ${W.concentration}. Of the ${inst.schedules_total.toLocaleString()} possible schedules, ${inst.schedules_valid} cover the shortfall exactly.`),
  table(["Feeder block", "Load (MW)", "Social cost per slot"],
    inst.feeders.map((fd) => [fd.name, fd.load * mw, f1(fd.social_cost)]), [4026, 2500, 2500]),
  spacer(),
  p("We measure five things for every schedule: the total objective, the social cost, how many slots the worst-hit feeder is off, how many back-to-back outages there are, and whether the shortfall is covered in every slot."),

  h1("3. Quantum approach and why we chose it"),
  h2("Why QAOA"),
  p("Our problem is a QUBO, and a QUBO maps directly onto an Ising Hamiltonian whose lowest-energy state is the best schedule. QAOA [2] is the standard gate-based algorithm for this kind of problem. It needs one qubit per variable, its circuit is built from the cost function itself, and the classical optimiser only has to tune two angles per layer. It has also been applied to power-system scheduling before, for example unit commitment [6]. We considered Grover search, but it would need a much deeper circuit to compute the cost inside the quantum computer. VQE with a general-purpose circuit would ignore the structure of the problem."),
  h2("Mapping to qubits"),
  p(["Substituting ", mono("x = (1 − Z) / 2"), " turns E(x) into ", mono("H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const"), `. A measured 1 on qubit (f,t) means "feeder f is off in slot t". We build h and J by hand and store H as a Qiskit SparsePauliOp. A unit test checks that the diagonal of this operator equals the classical objective for all ${inst.schedules_total.toLocaleString()} bitstrings.`]),
  h2("Two encodings"),
  p("Penalty QAOA, the textbook version, puts all four terms into H. The circuit starts in |+⟩ on every qubit and mixes with RX gates, so it can reach all 2ⁿ bitstrings, and most of those do not cover the shortfall."),
  p("Constraint-preserving QAOA is our main solver. Following the Quantum Alternating Operator Ansatz [3], it moves the hard rule out of the cost function and into the circuit. Each slot's qubits start in a Dicke state [4], an equal superposition of every way to switch off exactly D_t feeders. The mixer applies XY (XX+YY) rotations between qubits of the same slot [5]. Each rotation swaps a 1 and a 0, so the number of feeders off in that slot never changes. The cost Hamiltonian then only has to encode social cost and fairness."),
  p(`This removes the largest coefficients from H and shrinks the space the circuit explores from ${inst.schedules_total.toLocaleString()} states to ${inst.schedules_valid}, all of them valid. It costs two things: every feeder block must carry the same load, and the circuit is deeper (${cLast.two_qubit_gates} CX gates at p = ${cLast.p}, against ${pLast.two_qubit_gates} for the penalty version).`),
  ...figure("circuit_p1.png", "Figure 1. One QAOA layer on 12 qubits. Qubits 0 to 3, 4 to 7 and 8 to 11 are the three time slots. Each slot starts in a Dicke state, then passes through the cost layer (RZ and RZZ gates) and the XY mixers.", 600),

  h1("4. Architecture"),
  ...figure("architecture.png", "Figure 2. The FairShed pipeline. Blue boxes run in Qiskit, grey boxes are classical, and green boxes check the answer.", 600),
  p("Steps 3 and 4 run in Qiskit. Step 5 is the classical optimiser that tunes the circuit's angles. The exact baseline sits beside the pipeline, so every schedule the circuit produces can be checked against the true optimum."),

  h1("5. Implementation"),
  p("The circuit uses Qiskit's QuantumCircuit with a ParameterVector, StatePreparation for the Dicke states and XXPlusYYGate for the mixer. The Hamiltonian is a SparsePauliOp. Inside the optimisation loop we evaluate ⟨H⟩ with the StatevectorEstimator primitive. Sampling runs on Qiskit Aer, with and without a depolarizing noise model (0.1% error per one-qubit gate and 1% per CX)."),
  p(`COBYLA tunes the angles from several starting points: an annealing-style ramp, a warm start from the previous depth, and seeded random points. We keep the run with the lowest ⟨H⟩, measure the tuned circuit ${R.shots.toLocaleString()} times, treat each bitstring as a schedule, and return the lowest-cost valid one.`),
  p("All random seeds are fixed. One command, python run_experiments.py, regenerates every figure and the results file this report is built from, and a second run gave identical numbers. Twenty automated tests check the QUBO, the Ising mapping, the bit order, the claim that the constrained circuit only produces valid schedules, and the demo app end to end. Package versions are pinned in requirements.txt."),
  p("We also built a Streamlit app (app.py), live at fairshed.streamlit.app. An operator edits the feeders, the hourly shortfall and the fairness level, and gets back the schedule, a plain-language outage notice, and a side-by-side comparison with the exact optimum and the priority list. The quantum solver runs in its own process, because calling the Qiskit estimator from Streamlit's script thread crashed the server on our machine. The app accepts problems up to 15 qubits."),

  h1("6. Results"),
  h2("Schedules: priority list, exact optimum and quantum"),
  ...figure("schedules.png", "Figure 3. The priority list keeps one feeder dark all evening. The exact optimum rotates the outages, and the best QAOA sample is the same schedule.", 620),
  table(["", "Priority list (greedy)", "Exact optimum", `QAOA best sample (p = ${cLast.p})`],
    [
      ["Objective (lower is better)", f1(R.greedy.total), f1(R.exact.total), f1(R.qaoa_best_schedule.total)],
      ["Social cost", f1(R.greedy.social_cost), f1(R.exact.social_cost), f1(R.qaoa_best_schedule.social_cost)],
      ["Worst-hit feeder: slots off", R.greedy.max_outages, R.exact.max_outages, R.qaoa_best_schedule.max_outages],
      ["Back-to-back outages", R.greedy.consecutive_pairs, R.exact.consecutive_pairs, R.qaoa_best_schedule.consecutive_pairs],
      [`Rank among ${inst.schedules_total.toLocaleString()} schedules`, R.greedy.rank, R.exact.rank, R.qaoa_best_schedule.rank],
    ], [3026, 2000, 2000, 2000]),
  spacer(),
  p(`The priority list has the lowest social cost (${f1(R.greedy.social_cost)}) because it only cuts the cheapest feeders, but it leaves one feeder off for all ${R.greedy.max_outages} slots, with ${R.greedy.consecutive_pairs} back-to-back outages. The optimum accepts a social cost of ${f1(R.exact.social_cost)} to remove every back-to-back outage and bring the worst case down to ${R.exact.max_outages} slots. The weights B and C decide how much of that trade the operator wants.`),

  h2("Quantum performance against the exact reference"),
  p(`The exact reference is a brute-force enumeration of all ${inst.schedules_total.toLocaleString()} schedules. P(optimum) below is the probability that a single measurement returns the exact optimum, computed from the noise-free statevector. For comparison, picking one of the ${inst.schedules_valid} valid schedules at random succeeds ${pct(uniValid, 2)} of the time, and picking any bitstring at random ${pct(R.uniform_all_p_optimal, 3)} of the time.`),
  table(["Encoding", "p", "P(optimum)", "Valid shots", "Approx. ratio", "CX gates", `Best of ${R.shots} shots optimal`],
    R.qaoa.map((r) => [
      r.mode === "constrained" ? "Constraint-preserving" : "Penalty",
      r.p, pct(r.p_optimal, 2), pct(r.p_exact_cover, 1), r.approximation_ratio.toFixed(3),
      r.two_qubit_gates, r.sampled_best_is_optimal ? "Yes" : "No",
    ]), [2326, 500, 1300, 1200, 1300, 1000, 1400]),
  spacer(),
  p(`The approximation ratio is (E_max − ⟨E⟩) / (E_max − E_min) over all schedules, so 1.0 would mean every shot is optimal. At p = ${cLast.p} the constraint-preserving circuit measures the optimum about ${(cLast.p_optimal / uniValid).toFixed(0)} times as often as a random valid guess and about ${(cLast.p_optimal / pLast.p_optimal).toFixed(0)} times as often as penalty QAOA. Adding layers helps steadily: ${con.map((r) => pct(r.p_optimal, 1)).join(", ")} at p = ${con.map((r) => r.p).join(", ")}.`),
  ...figure("distribution.png", `Figure 4. Noise-free measurement distribution at p = ${cLast.p}. The exact optimum is the single most likely outcome.`, 520),
  ...figure("depth_sweep.png", "Figure 5. Effect of depth and of hardware-like noise. Left: probability of measuring the optimum. Right: probability that a shot satisfies the shortfall rule.", 620),
  p(`We also ran both circuits with depolarizing noise, and the results get worse. The constraint-preserving circuit still measured the optimum more often than the penalty version (${pct(cLast.noisy_p_optimal, 2)} against ${pct(pLast.noisy_p_optimal, 2)} at p = ${cLast.p}), but it lost its guarantee of valid output. The share of noisy shots that still covered the shortfall fell from ${con.map((r) => pct(r.noisy_p_exact_cover, 0)).join(" to ")} as we went from p = 1 to p = ${cLast.p}, while the noisy chance of the optimum stayed flat at about ${pct(cLast.noisy_p_optimal, 1)}. On an ideal simulator more layers help; once noise is added they start to hurt.`),

  h2("Cost versus fairness"),
  p("Scaling B and C together by one factor moves the optimum along the trade-off. For each setting we compared the exact best valid schedule with the best QAOA sample at p = 2."),
  table(["Fairness weight", "Social cost (exact)", "Outage slots per feeder (exact)", "Back-to-back", "QAOA finds it"],
    sweepRows, [1700, 1800, 2726, 1300, 1500]),
  spacer(),
  ...figure("tradeoff.png", "Figure 6. As fairness gets more weight, the optimal schedule costs more and the worst-hit feeder is off for fewer slots. Dots: best QAOA sample.", 600),
  p(`Feeder order in the table: ${inst.feeders.map((fd) => fd.name).join(" / ")}. With fairness switched off the answer is the priority list. At high fairness weights (4× and above) the model switches the hospital off for one slot so that every feeder takes one outage. A real operator would protect the hospital outright, which the app supports. That is why the operator sets the weights, and the algorithm only follows them.`),

  ...(big ? [
    h2("Scaling check: 15 qubits"),
    p(`We ran the same code on ${big.qubits} qubits (5 feeder blocks over 3 slots: ${big.schedules_total.toLocaleString()} schedules, ${big.schedules_valid.toLocaleString()} of them valid) with a smaller optimiser budget.`),
    table(["p", "P(optimum)", "Random valid guess", "Approx. ratio", "CX gates", `Best of ${R.shots} shots optimal`],
      big.rows.map((r) => [r.p, pct(r.p_optimal, 2), pct(1 / big.schedules_valid, 2), r.approximation_ratio.toFixed(3),
        r.two_qubit_gates, r.sampled_best_is_optimal ? "Yes" : "No"]), [700, 1500, 1900, 1600, 1300, 2026]),
    spacer(),
    p(`The chance of measuring the optimum is lower than at 12 qubits, which is expected with about ${Math.round(big.schedules_valid / inst.schedules_valid)} times as many valid schedules. It is still ${big.rows.map((r) => (r.p_optimal * big.schedules_valid).toFixed(0)).join(" and ")} times better than a random valid guess, and the best of ${R.shots.toLocaleString()} shots was the optimum at both depths.`),
  ] : []),

  h1("7. What worked"),
  bullet(`The quantum answer matched the exact optimum in every noise-free test we ran: the default case at p = ${con.map((r) => r.p).join(", ")}, ${R.tradeoff.filter((s) => s.qaoa_matches_exact).length} of ${R.tradeoff.length} fairness settings, and the 15-qubit case at both depths.`),
  bullet(`Building the constraint into the circuit made a measurable difference. At the same depth the optimum became about ${(cLast.p_optimal / pLast.p_optimal).toFixed(0)} times more likely than with penalty QAOA, and all noise-free shots were valid schedules instead of ${pct(pLast.p_exact_cover, 0)}.`),
  bullet("The fairness term can remember outages from earlier in the day. When a feeder has already been off, the optimum shifts its outages onto other feeders, and a test checks this."),
  bullet("The results are reproducible. Seeds are fixed, a second full run gave identical numbers, and twenty tests cover the model, the quantum mapping and the app."),
  bullet("It works end to end as an app an operator could try today, with an outage notice they could publish."),

  h1("8. Limitations"),
  bullet("There is no quantum advantage here. Brute force over 4,096 schedules takes milliseconds, and that is how we check the answer. In the app the quantum solver takes 15 to 20 seconds on a laptop for the same problem. A standard integer-programming solver would handle realistic sizes classically."),
  bullet("Everything ran on a simulator. Our noise model is a uniform depolarizing channel, which is simpler than a real device, and we assumed every qubit can interact with every other. Real hardware would need extra SWAP gates and would add more error."),
  bullet(`The optimum is found by sampling. One shot is the optimum ${pct(cLast.p_optimal, 1)} of the time at p = ${cLast.p}, and we report the best of ${R.shots.toLocaleString()} shots. With only ${inst.schedules_valid} valid schedules, ${R.shots.toLocaleString()} random valid guesses would almost certainly find the optimum too. The fair comparison is the per-shot probability, and that is the number we use.`),
  bullet("Feeder blocks must be the same size. The constraint-preserving circuit needs every block to carry the same load. Our QUBO handles unequal loads, but then only the penalty version applies, and it performs far worse at low depth."),
  bullet(`Noise breaks the built-in rule. Bit-flip errors push the state out of the valid set: at p = ${cLast.p} about ${pct(1 - cLast.noisy_p_exact_cover, 0)} of noisy shots break the shortfall rule and have to be thrown away.`),
  bullet("The Dicke-state preparation is not efficient. We used Qiskit's generic StatePreparation. The construction in [4] would need fewer CX gates."),
  bullet("The grid model is simplified. The social costs and weights are our own example values, not data from a distribution company. The model ignores network topology, ramp rates and minimum on and off times. The weights also need care: if A is too small, the penalty version prefers to leave part of the shortfall uncovered."),
  bullet("It does not scale to a real city yet. Qubits grow as feeders × slots, so hundreds of feeders over 24 hours is far beyond current devices, and tuning QAOA angles gets harder as problems grow. One option would be to split a city into zones and solve each separately. We have not tested that."),
  bullet("Our video says the quantum answer is always identical to the exact one. That is true for every noise-free test in this report, but it is not a general guarantee, and with noise the chance of hitting the optimum drops."),

  h1("9. Reproducing the results"),
  p([mono("git clone https://github.com/abzakir/FairShed.git")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("pip install -r requirements.txt")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("python -m pytest -q")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("python run_experiments.py")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("streamlit run app.py")], { para: { indent: { left: 360 } } }),
  p(`The full experiment run takes about ${Math.round(R.runtime_seconds / 60)} minutes on a laptop (Python 3.14, Qiskit 2.6.2, Qiskit Aer 0.18.2).`),

  h1("References"),
  p(["[1] K. Kiani, \"Power sector bleeds Rs1tr annually\", Dawn, 26 February 2026 (report on NEPRA's FY2024-25 performance evaluation of distribution companies). ", link("dawn.com/news/1975814", "https://www.dawn.com/news/1975814/power-sector-bleeds-rs1tr-annually")]),
  p("[2] E. Farhi, J. Goldstone, S. Gutmann, \"A Quantum Approximate Optimization Algorithm\", arXiv:1411.4028 (2014)."),
  p("[3] S. Hadfield et al., \"From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\", Algorithms 12(2), 34 (2019), arXiv:1709.03489."),
  p("[4] A. Bärtschi, S. Eidenbenz, \"Deterministic Preparation of Dicke States\", arXiv:1904.07358 (2019)."),
  p("[5] Z. Wang, N. Rubin, J. Dominy, E. Rieffel, \"XY mixers: analytical and numerical results for QAOA\", Phys. Rev. A 101, 012320 (2020), arXiv:1904.09314."),
  p("[6] S. Koretsky et al., \"Adapting Quantum Approximation Optimization Algorithm (QAOA) for Unit Commitment\", arXiv:2110.12624 (2021)."),
];

const doc = new Document({
  creator: "Runtime Terrors",
  title: "FairShed: a quantum scheduler for fair load shedding",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
      title: { run: { font: FONT, size: 56, bold: true, color: "0B0B0B" }, paragraph: { spacing: { before: 0, after: 60 }, keepNext: true } },
      heading1: { run: { font: FONT, size: 30, bold: true, color: "184F95" }, paragraph: { spacing: { before: 320, after: 120 }, keepNext: true } },
      heading2: { run: { font: FONT, size: 24, bold: true, color: "0B0B0B" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true } },
    },
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }],
    }],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "FairShed · page ", font: FONT, size: 18, color: "898781" }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "898781" })],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2] || path.join(__dirname, "FairShed_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
