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
  p([b("Team: "), "[team name and members]"], { para: { spacing: { after: 40 } } }),
  p([b("Code: "), link("github.com/abzakir/FairShed", "https://github.com/abzakir/FairShed")], { para: { spacing: { after: 40 } } }),
  p([b("Video: "), "[3-minute pitch and demo link]"], { para: { spacing: { after: 240 } } }),

  h1("Summary"),
  p(`When electricity supply falls short, a distribution company must decide which feeders to switch off in each hour. FairShed formulates that decision as a quadratic binary optimisation problem with two competing objectives, low social cost and fair sharing of outages, and solves it with the Quantum Approximate Optimization Algorithm (QAOA) in Qiskit.`),
  p([
    `On a ${inst.qubits}-qubit test instance (${inst.feeders.length} feeder blocks, ${inst.slots.length} evening hours), the best schedule measured from the quantum circuit is `,
    b(cLast.sampled_best_is_optimal ? "identical to the exact optimum" : "close to the exact optimum"),
    ` found by enumerating all ${inst.schedules_total.toLocaleString()} schedules. Our main technical result is that encoding the hard "cover the shortfall" rule into the circuit itself (Dicke-state initialisation and an XY mixer) raises the probability of measuring the optimum from ${pct(pLast.p_optimal, 2)} with textbook penalty QAOA to ${pct(cLast.p_optimal, 1)} at the same depth (p = ${cLast.p}), with 100% of noise-free shots valid.`,
  ]),
  p("We do not claim a quantum speedup. At this size a laptop solves the problem exactly in milliseconds; the contribution is a correct, verified quantum formulation and an honest analysis of where it works and where it breaks."),

  h1("1. Problem and motivation"),
  p("Load shedding in Pakistan is planned feeder by feeder, and the hours of outage on a feeder are tied to its revenue losses (theft and unpaid bills). In its performance evaluation of distribution companies for FY2024-25 the regulator NEPRA called these revenue-loss-based power cuts illegal and found that they had failed to deliver results [1]. Any fixed priority list also has a predictable failure: the same neighbourhoods lose power for several hours in a row while others are never touched."),
  p([b("Intended user: "), "the control-room operator of a distribution company who receives an hourly shortfall forecast and must publish a shedding schedule."]),
  p([b("The decision: "), "which feeder blocks to switch off in which hour so that the shortfall is covered, critical loads such as hospitals are protected, and no community carries the whole burden."]),
  p("The two objectives conflict. The cheapest schedule always cuts the same low-priority feeder; the fairest schedule rotates outages onto feeders whose outage costs more. That tension is what makes it an optimisation problem instead of a lookup."),

  h1("2. Optimisation formulation"),
  h2("Decision variables"),
  p([mono("x[f,t] = 1"), " if feeder block f is switched off in time slot t, otherwise 0. With F feeders and T slots there are F × T binary variables, one qubit each."]),
  h2("Objective (minimise)"),
  formula("E(x) = Σ c_f · x[f,t]                          social cost of the outages"),
  formula("     + B · Σ x[f,t] · x[f,t+1]                  back-to-back outages on one feeder"),
  formula("     + C · Σ_f ( h_f + Σ_t x[f,t] )²            outages concentrated on one feeder"),
  formula("     + A · Σ_t ( Σ_f P_f · x[f,t] − D_t )²      shortfall not covered exactly"),
  p("c_f is the social cost of one slot of outage on feeder f, P_f its load, D_t the shortfall in slot t, and h_f the number of slots feeder f was already off earlier today (0 in the test instance), so fairness remembers earlier outages. The squared concentration term is minimised, for a fixed number of outages, when they are spread evenly. Every term is at most quadratic, so the problem is a QUBO with no slack variables."),
  h2("Constraint"),
  p("Hard rule: in every slot the load switched off must equal the shortfall, Σ_f P_f · x[f,t] = D_t. In the penalty encoding this is the A-term above. In our main encoding it is enforced by the circuit and never violated (Section 3)."),
  h2("Test instance"),
  p(`${inst.feeders.length} feeder blocks of ${mw} MW each over ${inst.slots.length} evening-peak slots: ${inst.qubits} variables. Shortfall: ${inst.deficits.map((d, k) => `${d * mw} MW at ${inst.slots[k]}`).join(", ")}. Weights: A = ${W.deficit}, B = ${W.consecutive}, C = ${W.concentration}. Of the ${inst.schedules_total.toLocaleString()} possible schedules, ${inst.schedules_valid} cover the shortfall exactly.`),
  table(["Feeder block", "Load (MW)", "Social cost per slot"],
    inst.feeders.map((fd) => [fd.name, fd.load * mw, f1(fd.social_cost)]), [4026, 2500, 2500]),
  spacer(),
  p("Measurable outcomes: total objective, social cost, the number of outage slots on the worst-hit feeder, the number of back-to-back outages, and whether the shortfall is covered in every slot."),

  h1("3. Quantum approach and justification"),
  h2("Why QAOA"),
  p("The problem is a quadratic binary optimisation, which maps one-to-one onto an Ising Hamiltonian whose ground state is the optimal schedule. QAOA [2] is the standard gate-model algorithm for this class: it needs one qubit per variable, its circuit is built directly from the cost function, and it runs as a hybrid loop in which a classical optimiser tunes a small number of angles. It has already been studied for power-system scheduling problems such as unit commitment [6]. Grover-style search would need a far deeper arithmetic oracle, and VQE with a generic ansatz would ignore the problem structure."),
  h2("Mapping to qubits"),
  p(["Substituting ", mono("x = (1 − Z) / 2"), " turns E(x) into ", mono("H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const"), `. A measured 1 on qubit (f,t) means "feeder f off in slot t". We build h and J by hand, convert them to a Qiskit SparsePauliOp, and a unit test confirms that the diagonal of this operator equals the classical objective for all ${inst.schedules_total.toLocaleString()} bitstrings.`]),
  h2("Two encodings, compared"),
  bullet([b("Penalty QAOA (textbook). "), "All four terms go into H. The circuit starts in |+⟩ on every qubit and uses RX mixers, so it explores all 2ⁿ bitstrings, most of which do not cover the shortfall."]),
  bullet([b("Constraint-preserving QAOA (our main solver). "), "Following the Quantum Alternating Operator Ansatz [3], the hard rule is moved out of the cost function and into the circuit. Each slot's qubits start in a Dicke state [4], the equal superposition of all ways to switch off exactly D_t feeders. The mixer applies XY (XX+YY) rotations between qubits of the same slot [5], which exchange a 1 and a 0 and therefore keep the number of feeders off in that slot fixed. The cost Hamiltonian then only contains social cost and fairness."]),
  p(`The second encoding removes the largest coefficients from the Hamiltonian, shrinks the reachable space from ${inst.schedules_total.toLocaleString()} to ${inst.schedules_valid} states, and guarantees valid output on an ideal device. Its price is the assumption that feeder blocks have equal load, and a deeper circuit (${cLast.two_qubit_gates} CX gates at p = ${cLast.p}, against ${pLast.two_qubit_gates} for the penalty version).`),
  ...figure("circuit_p1.png", "Figure 1. One QAOA layer on 12 qubits. Qubits 0–3, 4–7 and 8–11 are the three time slots: Dicke-state preparation, cost layer (RZ and RZZ), then XY mixers inside each slot.", 600),

  h1("4. Architecture"),
  ...figure("architecture.png", "Figure 2. FairShed pipeline. Blue boxes run in Qiskit; green boxes are the validation path.", 600),

  h1("5. Implementation"),
  bullet([b("Qiskit components. "), "QuantumCircuit with ParameterVector, StatePreparation and XXPlusYYGate; SparsePauliOp for the Hamiltonian; the StatevectorEstimator primitive for ⟨H⟩ inside the optimisation loop; Statevector for the noise-free outcome distribution; Qiskit Aer for shot sampling with and without a depolarizing noise model (0.1% per one-qubit gate, 1% per CX)."]),
  bullet([b("Classical loop. "), "COBYLA from several starting points (an annealing-style ramp, a warm start from the previous depth, and seeded random points); the lowest ⟨H⟩ is kept."]),
  bullet([b("Decoding. "), `The tuned circuit is measured ${R.shots} times. Each bitstring is a schedule; the lowest-cost valid one is returned and checked against every constraint.`]),
  bullet([b("Reproducibility. "), "All seeds are fixed. One command, python run_experiments.py, regenerates every figure and the results file this report is built from. Twenty automated tests cover the QUBO, the Ising mapping, the bit-order convention, the constraint-preserving property, and the demo app end to end. Package versions are pinned in requirements.txt."]),

  bullet([b("Demo application. "), "A Streamlit app (app.py) lets an operator edit the feeders, the hourly shortfall and the fairness level, then returns the schedule, a plain-language outage notice and a comparison with the exact optimum and the priority list. The quantum solver runs in a separate worker process, because calling Qiskit's estimator from the web server's own thread crashed it on our platform; problems up to 15 qubits are supported."]),

  h1("6. Results"),
  h2("Schedules: greedy, exact and quantum"),
  ...figure("schedules.png", "Figure 3. The greedy priority list keeps one feeder dark all evening. The exact optimum rotates the outages. The best QAOA sample reproduces the exact optimum.", 620),
  table(["", "Greedy priority list", "Exact optimum", `QAOA best sample (p = ${cLast.p})`],
    [
      ["Objective (lower is better)", f1(R.greedy.total), f1(R.exact.total), f1(R.qaoa_best_schedule.total)],
      ["Social cost", f1(R.greedy.social_cost), f1(R.exact.social_cost), f1(R.qaoa_best_schedule.social_cost)],
      ["Worst-hit feeder: slots off", R.greedy.max_outages, R.exact.max_outages, R.qaoa_best_schedule.max_outages],
      ["Back-to-back outages", R.greedy.consecutive_pairs, R.exact.consecutive_pairs, R.qaoa_best_schedule.consecutive_pairs],
      [`Rank among ${inst.schedules_total.toLocaleString()} schedules`, R.greedy.rank, R.exact.rank, R.qaoa_best_schedule.rank],
    ], [3026, 2000, 2000, 2000]),
  spacer(),
  p(`The greedy list has the lowest social cost (${f1(R.greedy.social_cost)}) because it only ever cuts the cheapest feeders, but it leaves one feeder off for ${R.greedy.max_outages} consecutive slots. The optimum accepts a social cost of ${f1(R.exact.social_cost)} to remove every back-to-back outage. This is the trade-off the weights B and C control.`),

  h2("Quantum performance against the exact reference"),
  p(`The exact reference is a brute-force enumeration of all ${inst.schedules_total.toLocaleString()} schedules. "P(optimum)" is the probability that a single measurement returns the exact optimum, computed from the noise-free statevector. For comparison, picking one of the ${inst.schedules_valid} valid schedules at random succeeds with probability ${pct(uniValid, 2)}, and picking any bitstring at random with ${pct(R.uniform_all_p_optimal, 3)}.`),
  table(["Encoding", "p", "P(optimum)", "Valid shots", "Approx. ratio", "CX gates", `Best of ${R.shots} shots optimal`],
    R.qaoa.map((r) => [
      r.mode === "constrained" ? "Constraint-preserving" : "Penalty",
      r.p, pct(r.p_optimal, 2), pct(r.p_exact_cover, 1), r.approximation_ratio.toFixed(3),
      r.two_qubit_gates, r.sampled_best_is_optimal ? "Yes" : "No",
    ]), [2326, 500, 1300, 1200, 1300, 1000, 1400]),
  spacer(),
  p(`The approximation ratio is (E_max − ⟨E⟩) / (E_max − E_min) over all schedules, so 1.0 means every shot is optimal. At p = ${cLast.p} the constraint-preserving circuit measures the optimum ${(cLast.p_optimal / uniValid).toFixed(0)} times more often than random valid guessing and ${(cLast.p_optimal / pLast.p_optimal).toFixed(0)} times more often than penalty QAOA.`),
  ...figure("distribution.png", `Figure 4. Noise-free measurement distribution at p = ${cLast.p}. The exact optimum is the single most likely outcome.`, 520),
  ...figure("depth_sweep.png", "Figure 5. Effect of depth and of hardware-like noise. Left: probability of measuring the optimum. Right: probability that a shot satisfies the shortfall rule.", 620),
  p(`Noise changes the picture. With depolarizing errors the constraint-preserving circuit still beats penalty QAOA on P(optimum) (${pct(cLast.noisy_p_optimal, 2)} against ${pct(pLast.noisy_p_optimal, 2)} at p = ${cLast.p}), but its guarantee of valid output is lost: only ${pct(cLast.noisy_p_exact_cover, 0)} of noisy shots still cover the shortfall exactly at p = ${cLast.p}, and adding layers no longer helps. More depth buys expressivity and costs fidelity.`),

  h2("Cost versus fairness"),
  p("Scaling both fairness weights B and C by a common factor moves the optimum along the trade-off. For each setting we compare the exact best valid schedule with the best QAOA sample (p = 2)."),
  table(["Fairness weight", "Social cost (exact)", "Outage slots per feeder (exact)", "Back-to-back", "QAOA finds it"],
    sweepRows, [1700, 1800, 2726, 1300, 1500]),
  spacer(),
  ...figure("tradeoff.png", "Figure 6. As fairness is weighted more heavily the optimal schedule costs more and the worst-hit feeder is off for fewer slots. Dots: best QAOA sample.", 600),
  p(`Feeder order in the table: ${inst.feeders.map((fd) => fd.name).join(" / ")}. With no fairness term the answer is the greedy schedule. At very high fairness weights the model starts to switch off the hospital feeder to equalise the burden, which shows that the weights are a policy choice and must be set by the operator, not by the algorithm.`),

  ...(big ? [
    h2("Scaling check: 15 qubits"),
    p(`The same code was run on ${big.qubits} qubits (5 feeder blocks, 3 slots; ${big.schedules_total.toLocaleString()} schedules, ${big.schedules_valid.toLocaleString()} valid) with a reduced optimiser budget.`),
    table(["p", "P(optimum)", "Random valid guess", "Approx. ratio", "CX gates", `Best of ${R.shots} shots optimal`],
      big.rows.map((r) => [r.p, pct(r.p_optimal, 2), pct(1 / big.schedules_valid, 2), r.approximation_ratio.toFixed(3),
        r.two_qubit_gates, r.sampled_best_is_optimal ? "Yes" : "No"]), [700, 1500, 1900, 1600, 1300, 2026]),
    spacer(),
  ] : []),

  h1("7. Limitations"),
  bullet([b("No quantum advantage. "), `Brute force over ${inst.schedules_total.toLocaleString()} schedules takes milliseconds and is how we verify the answer. A commercial integer-programming solver handles realistic instances of this problem classically. Nothing here shows a speedup.`]),
  bullet([b("Simulator only. "), "All results come from Qiskit's statevector simulator and Aer. The noise model is a uniform depolarizing channel, simpler than a real device, and assumes all-to-all connectivity; routing on real hardware would add SWAP gates."]),
  bullet([b("The optimum is found by sampling, not with certainty. "), `A single shot is optimal ${pct(cLast.p_optimal, 1)} of the time at p = ${cLast.p}. We report the best of ${R.shots} shots. On a ${inst.schedules_valid}-state valid space that many shots would also find the optimum by random valid guessing, so the fair claim is the per-shot amplification, not the best-of-shots result.`]),
  bullet([b("Equal-size blocks. "), "The constraint-preserving circuit needs every feeder block to carry the same load. Unequal feeders fall back to the penalty encoding, which performs far worse at low depth. Our QUBO supports unequal loads; the improved circuit does not."]),
  bullet([b("Noise breaks the hard constraint. "), "Bit-flip errors move the state out of the valid subspace. Invalid shots can be discarded, at the cost of more shots."]),
  bullet([b("State preparation cost. "), "Dicke states are prepared with Qiskit's generic StatePreparation, which is not gate-optimal. The construction of [4] would reduce the CX count."]),
  bullet([b("Simplified grid model. "), "Social costs and fairness weights are illustrative inputs, not calibrated data. The model ignores network topology, ramping, and minimum on/off durations, and the weights A, B, C must be tuned: if A is too small the penalty formulation prefers to ignore the shortfall."]),
  bullet([b("Scaling. "), "Qubits grow as feeders × slots. A city with hundreds of feeders and 24 slots is far beyond current devices, and QAOA angle optimisation becomes harder with size."]),

  h1("8. Reproducing the results"),
  p([mono("pip install -r requirements.txt")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("python -m pytest -q")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("python run_experiments.py")], { para: { spacing: { after: 40 }, indent: { left: 360 } } }),
  p([mono("streamlit run app.py")], { para: { indent: { left: 360 } } }),
  p(`The full experiment run took ${Math.round(R.runtime_seconds / 60)} minutes on a laptop (Python 3.14, Qiskit 2.6.2, Qiskit Aer 0.18.2).`),

  h1("References"),
  p(["[1] K. Kiani, \"Power sector bleeds Rs1tr annually\", Dawn, 26 February 2026 (report on NEPRA's FY2024-25 performance evaluation of distribution companies). ", link("dawn.com/news/1975814", "https://www.dawn.com/news/1975814/power-sector-bleeds-rs1tr-annually")]),
  p("[2] E. Farhi, J. Goldstone, S. Gutmann, \"A Quantum Approximate Optimization Algorithm\", arXiv:1411.4028 (2014)."),
  p("[3] S. Hadfield et al., \"From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\", Algorithms 12(2), 34 (2019), arXiv:1709.03489."),
  p("[4] A. Bärtschi, S. Eidenbenz, \"Deterministic Preparation of Dicke States\", arXiv:1904.07358 (2019)."),
  p("[5] Z. Wang, N. Rubin, J. Dominy, E. Rieffel, \"XY mixers: analytical and numerical results for QAOA\", Phys. Rev. A 101, 012320 (2020), arXiv:1904.09314."),
  p("[6] S. Koretsky et al., \"Adapting Quantum Approximation Optimization Algorithm (QAOA) for Unit Commitment\", arXiv:2110.12624 (2021)."),
];

const doc = new Document({
  creator: "FairShed team",
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
  const out = path.join(__dirname, "FairShed_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
