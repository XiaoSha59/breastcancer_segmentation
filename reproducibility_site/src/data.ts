export type Metric = {
  label: string;
  value: string;
  detail: string;
  tone: "teal" | "blue" | "indigo" | "coral";
};

export type PipelineStep = {
  title: string;
  subtitle: string;
};

export type LogLine = {
  text: string;
  kind?: "info" | "pass" | "warn";
  step: number;
  pause?: number;
};

export type CaseExample = {
  id: string;
  cohort: string;
  title: string;
  tag: string;
  rawDice: number;
  lccDice: number;
  explanation: string;
  image: string;
  sourcePath: string;
  tone: "good" | "bad";
};

export const metrics: Metric[] = [
  {
    label: "Internal Dice",
    value: "0.786",
    detail: "Selected residual encoder nnU-Net",
    tone: "teal",
  },
  {
    label: "Zero-shot external",
    value: "0.547",
    detail: "Raw prediction, baseline phase selection",
    tone: "blue",
  },
  {
    label: "Timing-aware + LCC",
    value: "0.574",
    detail: "Primary external evaluation setting",
    tone: "indigo",
  },
  {
    label: "LCC-collapse",
    value: "43 / 526",
    detail: "Canonical postprocessing failure audit",
    tone: "coral",
  },
];

export const pipelineSteps: PipelineStep[] = [
  {
    title: "Cohort Construction",
    subtitle: "Audit overlap and retain independent external cases",
  },
  {
    title: "Phase Selection",
    subtitle: "Select DCE phases using acquisition timing",
  },
  {
    title: "External Evaluation",
    subtitle: "Run zero-shot evaluation on Duke, I-SPY1 and NACT-Pilot",
  },
  {
    title: "LCC Postprocessing",
    subtitle: "Retain the largest connected prediction component",
  },
  {
    title: "Failure Attribution",
    subtitle: "Measure patient-level errors and failure signatures",
  },
  {
    title: "Canonical Audit",
    subtitle: "Verify the manuscript-facing counts and statistics",
  },
];

export const runLog: LogLine[] = [
  { text: "[INFO] Starting reproducibility audit...", kind: "info", step: 0, pause: 250 },
  { text: "[INFO] MAMA-MIA cases: 1506", kind: "info", step: 0 },
  { text: "[INFO] BreastDCEDL / I-SPY2 development roster: 982", kind: "info", step: 0 },
  { text: "[INFO] Identifier-matched overlap removed: 980", kind: "info", step: 0 },
  { text: "[INFO] External cohort retained: 526", kind: "info", step: 0 },
  { text: "[INFO] Duke: 291 | I-SPY1: 171 | NACT-Pilot: 64", kind: "info", step: 0 },
  { text: "[PASS] ALL COHORT-CONSTRUCTION CHECKS PASSED", kind: "pass", step: 1, pause: 500 },
  { text: "[INFO] Loading timing-aware phase-selection summary...", kind: "info", step: 1 },
  { text: "[PASS] Phase metadata: 513 timed + 13 acquisition-order fallback", kind: "pass", step: 2, pause: 350 },
  { text: "[INFO] External raw Dice: 0.555", kind: "info", step: 2 },
  { text: "[INFO] Timing-aware + LCC Dice: 0.574", kind: "info", step: 3, pause: 350 },
  { text: "[INFO] LCC-collapse cases: 43 / 526", kind: "info", step: 4 },
  { text: "[INFO] Opposite-image-half flags: 54 / 526", kind: "info", step: 4 },
  { text: "[INFO] Collapse ∩ opposite-half: 30 / 43", kind: "info", step: 4 },
  { text: "[INFO] Odds ratio: 44.13 | Fisher p: 3.95e-24", kind: "info", step: 5, pause: 450 },
  { text: "[PASS] ALL MANUSCRIPT CHECKS PASSED", kind: "pass", step: 6, pause: 700 },
];

export const cases: CaseExample[] = [
  {
    id: "ISPY1_1128",
    cohort: "I-SPY1",
    title: "Representative case",
    tag: "LCC helps",
    rawDice: 0.3814842659,
    lccDice: 0.8169273229,
    explanation:
      "The raw prediction contains additional disconnected regions. Retaining the largest connected component substantially improves agreement with the reference segmentation.",
    image:
      "https://raw.githubusercontent.com/Helen2813/MRI-Project/main/12_media_orthogonal_figures/media_typical_case_orthogonal.png",
    sourcePath:
      "12_media_orthogonal_figures/media_typical_case_orthogonal.png",
    tone: "good",
  },
  {
    id: "DUKE_883",
    cohort: "Duke",
    title: "Postprocessing failure",
    tag: "LCC collapse",
    rawDice: 0.5224655502,
    lccDice: 0.0,
    explanation:
      "The raw prediction has meaningful overlap with the reference mask, but LCC selection retains a disconnected component instead. The case meets the canonical collapse definition.",
    image:
      "https://raw.githubusercontent.com/Helen2813/MRI-Project/main/12_media_orthogonal_figures/media_lcc_collapse_case_orthogonal.png",
    sourcePath:
      "12_media_orthogonal_figures/media_lcc_collapse_case_orthogonal.png",
    tone: "bad",
  },
];

export const cohortCounts = [
  { name: "Duke", value: 291 },
  { name: "I-SPY1", value: 171 },
  { name: "NACT-Pilot", value: 64 },
];

export const codeMap = [
  {
    stage: "Cohort construction",
    script: "reproducibility/build_external_cohort.py",
    output: "526-case external manifest",
  },
  {
    stage: "Phase selection",
    script: "06_external_validation/preprocess_mama_mia_nnunet_v2.py",
    output: "Timing-aware five-channel inputs",
  },
  {
    stage: "External metrics",
    script: "06_external_validation/analyze_external_phase_subset_home.py",
    output: "Cohort and timing summaries",
  },
  {
    stage: "Canonical audit",
    script:
      "15_final_lcc_collapse_profile/15b_canonical_lcc_flag_repair/run_15b_canonical_lcc_flag_repair.py",
    output: "Collapse / opposite-half audit",
  },
];
