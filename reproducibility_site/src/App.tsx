import { useEffect, useMemo, useRef, useState } from "react";
import {
  cases,
  codeMap,
  cohortCounts,
  metrics,
  pipelineSteps,
  runLog,
  type CaseExample,
} from "./data";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function Icon({ name }: { name: "run" | "image" | "code" | "check" | "zoom" | "reset" }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  if (name === "run") {
    return (
      <svg {...common}>
        <path d="M7 4v16l13-8L7 4Z" />
      </svg>
    );
  }
  if (name === "image") {
    return (
      <svg {...common}>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <circle cx="8.5" cy="9" r="1.5" />
        <path d="m4.5 17 5-5 3.5 3 2.5-2.5 4 4" />
      </svg>
    );
  }
  if (name === "code") {
    return (
      <svg {...common}>
        <path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" />
      </svg>
    );
  }
  if (name === "zoom") {
    return (
      <svg {...common}>
        <circle cx="11" cy="11" r="6" />
        <path d="m16 16 4 4M11 8v6M8 11h6" />
      </svg>
    );
  }
  if (name === "reset") {
    return (
      <svg {...common}>
        <path d="M20 6v5h-5" />
        <path d="M18.6 15a8 8 0 1 1 .4-7.9L20 11" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <path d="m5 12 4 4L19 6" />
    </svg>
  );
}

function MetricCard({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-card__head">
        <span>{label}</span>
        <div className="micro-bars" aria-hidden="true">
          <i />
          <i />
          <i />
          <i />
          <i />
        </div>
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function Pipeline({
  currentStep,
  done,
  running,
  elapsed,
}: {
  currentStep: number;
  done: boolean;
  running: boolean;
  elapsed: number;
}) {
  const progress =
    done ? 100 : Math.max(0, Math.min(100, (currentStep / (pipelineSteps.length - 1)) * 100));

  return (
    <section className="panel pipeline-panel">
      <div className="section-title-row">
        <div>
          <div className="eyebrow">EXECUTION TRACE</div>
          <h2>Run Pipeline</h2>
          <p>From leakage-controlled cohort construction to the canonical failure audit.</p>
        </div>
        <div className="run-meta">
          <span className={`status-pill ${done ? "verified" : running ? "running" : "ready"}`}>
            <i />
            {done ? "Verified" : running ? "Running" : "Ready"}
          </span>
          <span className="timer">{formatTime(elapsed)}</span>
        </div>
      </div>

      <div className="pipeline">
        <div className="pipeline-track">
          <div className="pipeline-progress" style={{ width: `${progress}%` }} />
        </div>

        {pipelineSteps.map((step, index) => {
          const complete = done || index < currentStep;
          const active = !done && running && index === currentStep;
          return (
            <div
              className={`pipeline-step ${complete ? "complete" : ""} ${active ? "active" : ""}`}
              key={step.title}
            >
              <div className="step-marker">
                {complete ? <Icon name="check" /> : <span>{index + 1}</span>}
              </div>
              <h3>{step.title}</h3>
              <p>{step.subtitle}</p>
              <small>{complete ? "COMPLETE" : active ? "RUNNING" : "WAITING"}</small>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function LiveLog({
  visibleLines,
  typingText,
  done,
}: {
  visibleLines: number;
  typingText: string;
  done: boolean;
}) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [visibleLines, typingText]);

  return (
    <section className="panel log-panel">
      <div className="section-title-row compact">
        <div>
          <div className="eyebrow">LIVE OUTPUT</div>
          <h2>Execution Log</h2>
        </div>
        <span className={`streaming ${done ? "verified" : ""}`}>
          <i />
          {done ? "Complete" : "Streaming"}
        </span>
      </div>
      <div className="terminal" ref={logRef}>
        {runLog.slice(0, visibleLines).map((line, idx) => (
          <div className={`terminal-line ${line.kind ?? "info"}`} key={`${line.text}-${idx}`}>
            {line.text}
          </div>
        ))}
        {!done && typingText && (
          <div className={`terminal-line ${runLog[visibleLines]?.kind ?? "info"}`}>
            {typingText}
            <span className="cursor" />
          </div>
        )}
        {done && <div className="terminal-prompt">audit@reproducibility:~$ <span className="cursor" /></div>}
      </div>
    </section>
  );
}

function CaseCard({
  item,
  onOpen,
}: {
  item: CaseExample;
  onOpen: (item: CaseExample) => void;
}) {
  const delta = item.lccDice - item.rawDice;
  return (
    <article className={`case-card ${item.tone}`}>
      <div className="case-card__top">
        <div>
          <span className="case-tag">{item.tag}</span>
          <h3>{item.title}</h3>
        </div>
        <button className="icon-button" onClick={() => onOpen(item)} aria-label={`Zoom ${item.id}`}>
          <Icon name="zoom" />
        </button>
      </div>

      <button className="image-button" onClick={() => onOpen(item)}>
        <img src={item.image} alt={`${item.id} DCE-MRI comparison panel`} />
        <span className="image-hover">Open image viewer</span>
      </button>

      <div className="case-stats">
        <span>{item.cohort}</span>
        <span>{item.id}</span>
        <strong>
          Dice {item.rawDice.toFixed(3)} → {item.lccDice.toFixed(3)}
          <em className={delta >= 0 ? "positive" : "negative"}>
            {delta >= 0 ? " +" : " "}
            {delta.toFixed(3)}
          </em>
        </strong>
      </div>
      <p>{item.explanation}</p>
      <small>Source panel: {item.sourcePath}</small>
    </article>
  );
}

function ImageModal({
  item,
  onClose,
}: {
  item: CaseExample | null;
  onClose: () => void;
}) {
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    setZoom(1);
  }, [item]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!item) return null;

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" onMouseDown={onClose}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <span className="case-tag">{item.tag}</span>
            <h2>{item.cohort} · {item.id}</h2>
            <p>
              Raw Dice {item.rawDice.toFixed(3)} → LCC Dice {item.lccDice.toFixed(3)}
            </p>
          </div>
          <div className="modal-actions">
            <button onClick={() => setZoom((z) => Math.max(0.7, z - 0.15))}>−</button>
            <span>{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom((z) => Math.min(2.6, z + 0.15))}>+</button>
            <button className="close-button" onClick={onClose}>Close</button>
          </div>
        </div>

        <div className="modal-image-viewport">
          <img
            src={item.image}
            alt={`${item.id} enlarged DCE-MRI comparison`}
            style={{ transform: `scale(${zoom})` }}
          />
        </div>

        <div className="modal-note">
          <strong>What this panel shows</strong>
          <p>{item.explanation}</p>
          <span>Use + / − to inspect the segmentation overlays. Press Esc to close.</span>
        </div>
      </div>
    </div>
  );
}

function CohortSummary() {
  const max = Math.max(...cohortCounts.map((d) => d.value));
  return (
    <section className="panel cohort-panel">
      <div className="section-title-row compact">
        <div>
          <div className="eyebrow">EXTERNAL COHORT</div>
          <h2>526 cases after overlap removal</h2>
        </div>
        <strong className="cohort-total">526</strong>
      </div>
      <div className="cohort-grid">
        {cohortCounts.map((item, index) => (
          <div className="cohort-row" key={item.name}>
            <div className="cohort-label">
              <span>{item.name}</span>
              <strong>{item.value}</strong>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill bar-${index}`}
                style={{ width: `${(item.value / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function CodeMap() {
  return (
    <section className="panel code-panel">
      <div className="section-title-row compact">
        <div>
          <div className="eyebrow">TRACEABILITY</div>
          <h2>Manuscript → Code</h2>
        </div>
        <a
          className="text-link"
          href="https://github.com/Helen2813/MRI-Project"
          target="_blank"
          rel="noreferrer"
        >
          View repository ↗
        </a>
      </div>
      <div className="code-list">
        {codeMap.map((row) => (
          <div className="code-row" key={row.stage}>
            <span>{row.stage}</span>
            <code>{row.script}</code>
            <small>{row.output}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatTime(seconds: number) {
  const mm = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const ss = (seconds % 60).toString().padStart(2, "0");
  return `${mm}:${ss}`;
}

export default function App() {
  const [runKey, setRunKey] = useState(0);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [visibleLines, setVisibleLines] = useState(0);
  const [typingText, setTypingText] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [selectedCase, setSelectedCase] = useState<CaseExample | null>(null);

  const runDemo = () => {
    setRunKey((value) => value + 1);
  };

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const execute = async () => {
      setRunning(true);
      setDone(false);
      setCurrentStep(0);
      setVisibleLines(0);
      setTypingText("");
      setElapsed(0);

      timer = window.setInterval(() => {
        setElapsed((value) => value + 1);
      }, 1000);

      await sleep(450);
      for (let lineIndex = 0; lineIndex < runLog.length && !cancelled; lineIndex += 1) {
        const line = runLog[lineIndex];
        setCurrentStep(Math.min(line.step, pipelineSteps.length - 1));
        setTypingText("");

        for (let i = 1; i <= line.text.length && !cancelled; i += 1) {
          setTypingText(line.text.slice(0, i));
          await sleep(i < 8 ? 12 : 7);
        }

        if (cancelled) break;
        setVisibleLines(lineIndex + 1);
        setTypingText("");
        await sleep(line.pause ?? 90);
      }

      if (!cancelled) {
        setCurrentStep(pipelineSteps.length - 1);
        setRunning(false);
        setDone(true);
        if (timer) window.clearInterval(timer);
      }
    };

    execute();

    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [runKey]);

  const runStatus = useMemo(() => {
    if (done) return "All checks passed";
    if (running) return pipelineSteps[currentStep]?.title ?? "Running";
    return "Ready";
  }, [done, running, currentStep]);

  return (
    <>
      <main className="app-shell">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div>
              <h1>Breast DCE-MRI Reliability</h1>
              <p>
                Research reproducibility companion
                <span>Not clinical software</span>
              </p>
            </div>
          </div>

          <div className="header-actions">
            <div className="run-state">
              <small>Current state</small>
              <strong>{runStatus}</strong>
            </div>
            <button className="primary-button" onClick={runDemo}>
              <Icon name="reset" />
              Replay run
            </button>
          </div>
        </header>

        <section className="metric-grid">
          {metrics.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </section>

        <Pipeline
          currentStep={currentStep}
          done={done}
          running={running}
          elapsed={elapsed}
        />

        <section className="main-grid">
          <div className="left-stack">
            <section className="panel cases-panel">
              <div className="section-title-row">
                <div>
                  <div className="eyebrow">PATIENT-LEVEL EXAMPLES</div>
                  <h2>Representative MRI Cases</h2>
                  <p>
                    Published orthogonal panels used to illustrate when LCC helps and when it fails.
                  </p>
                </div>
                <div className="legend">
                  <span><i className="legend-expert" />Reference</span>
                  <span><i className="legend-raw" />Raw</span>
                  <span><i className="legend-lcc" />LCC</span>
                </div>
              </div>

              <div className="case-grid">
                {cases.map((item) => (
                  <CaseCard item={item} onOpen={setSelectedCase} key={item.id} />
                ))}
              </div>
            </section>

            <CohortSummary />
          </div>

          <div className="right-stack">
            <LiveLog visibleLines={visibleLines} typingText={typingText} done={done} />
            <CodeMap />
          </div>
        </section>

        <footer>
          <span>Cross-cohort segmentation reliability · External validation · Failure attribution</span>
          <span>Public research demo</span>
        </footer>
      </main>

      <ImageModal item={selectedCase} onClose={() => setSelectedCase(null)} />
    </>
  );
}
