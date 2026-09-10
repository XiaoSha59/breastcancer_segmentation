import { useEffect, useMemo, useRef, useState } from "react";
import {
  cases,
  codeMap,
  cohortCounts,
  metrics,
  pipelineSteps,
  type CaseExample,
} from "./data";
import {
  API_BASE,
  cancelAudit,
  checkBackend,
  startAudit,
  subscribeToRun,
  type BackendEvent,
} from "./api";
import "./backend.css";

type LogEntry = {
  text: string;
  kind: "info" | "pass" | "warn" | "error";
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function Icon({ name }: { name: "check" | "zoom" | "run" | "stop" }) {
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

  if (name === "zoom") {
    return (
      <svg {...common}>
        <circle cx="11" cy="11" r="6" />
        <path d="m16 16 4 4M11 8v6M8 11h6" />
      </svg>
    );
  }

  if (name === "run") {
    return (
      <svg {...common}>
        <path d="M7 4v16l13-8L7 4Z" />
      </svg>
    );
  }

  if (name === "stop") {
    return (
      <svg {...common}>
        <rect x="6" y="6" width="12" height="12" rx="2" />
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
          <i /><i /><i /><i /><i />
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
  failed,
  elapsed,
}: {
  currentStep: number;
  done: boolean;
  running: boolean;
  failed: boolean;
  elapsed: number;
}) {
  const progress = done
    ? 100
    : Math.max(0, Math.min(100, (currentStep / (pipelineSteps.length - 1)) * 100));

  return (
    <section className="panel pipeline-panel">
      <div className="section-title-row">
        <div>
          <div className="eyebrow">REAL REPRODUCIBILITY AUDIT</div>
          <h2>Run Pipeline</h2>
          <p>
            The backend validates the committed research artifacts stage by stage.
            GPU training and inference are not rerun in this lightweight audit.
          </p>
        </div>
        <div className="run-meta">
          <span
            className={`status-pill ${
              failed ? "failed" : done ? "verified" : running ? "running" : "ready"
            }`}
          >
            <i />
            {failed ? "Failed" : done ? "Verified" : running ? "Running" : "Ready"}
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
              className={`pipeline-step ${complete ? "complete" : ""} ${
                active ? "active" : ""
              }`}
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
  logs,
  typing,
  done,
  failed,
}: {
  logs: LogEntry[];
  typing: LogEntry | null;
  done: boolean;
  failed: boolean;
}) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [logs, typing]);

  return (
    <section className="panel log-panel">
      <div className="section-title-row compact">
        <div>
          <div className="eyebrow">STREAMED FROM PYTHON</div>
          <h2>Execution Log</h2>
        </div>
        <span className={`streaming ${done ? "verified" : failed ? "failed" : ""}`}>
          <i />
          {failed ? "Stopped" : done ? "Complete" : "Live"}
        </span>
      </div>
      <div className="terminal" ref={logRef}>
        {logs.map((line, idx) => (
          <div className={`terminal-line ${line.kind}`} key={`${idx}-${line.text}`}>
            {line.text || "\u00a0"}
          </div>
        ))}
        {typing && (
          <div className={`terminal-line ${typing.kind}`}>
            {typing.text}<span className="cursor" />
          </div>
        )}
        {done && (
          <div className="terminal-prompt">
            audit@reproducibility:~$ <span className="cursor" />
          </div>
        )}
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
        <button
          className="icon-button"
          onClick={() => onOpen(item)}
          aria-label={`Zoom ${item.id}`}
        >
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

  useEffect(() => setZoom(1), [item]);

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
      <div className="modal" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <span className="case-tag">{item.tag}</span>
            <h2>{item.cohort} · {item.id}</h2>
            <p>
              Raw Dice {item.rawDice.toFixed(3)} → LCC Dice {item.lccDice.toFixed(3)}
            </p>
          </div>
          <div className="modal-actions">
            <button onClick={() => setZoom((value) => Math.max(0.7, value - 0.15))}>−</button>
            <span>{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom((value) => Math.min(2.6, value + 0.15))}>+</button>
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
  const max = Math.max(...cohortCounts.map((item) => item.value));

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
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const rest = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [backendChecked, setBackendChecked] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [failed, setFailed] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [selectedCase, setSelectedCase] = useState<CaseExample | null>(null);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [typing, setTyping] = useState<LogEntry | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [backendFinished, setBackendFinished] = useState(false);

  const logQueue = useRef<LogEntry[]>([]);
  const logTyperBusy = useRef(false);
  const generation = useRef(0);
  const unsubscribeRef = useRef<null | (() => void)>(null);

  const refreshHealth = async () => {
    try {
      const health = await checkBackend();
      setBackendOnline(Boolean(health.ok));
    } catch {
      setBackendOnline(false);
    } finally {
      setBackendChecked(true);
    }
  };

  useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      setElapsed((value) => value + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (backendFinished && pendingCount === 0 && !typing) {
      setRunning(false);
      setDone(!failed);
    }
  }, [backendFinished, pendingCount, typing, failed]);

  const drainLogs = async (localGeneration: number) => {
    if (logTyperBusy.current) return;
    logTyperBusy.current = true;

    while (logQueue.current.length > 0 && generation.current === localGeneration) {
      const next = logQueue.current.shift()!;
      setPendingCount((count) => Math.max(0, count - 1));

      // Empty lines are displayed immediately.
      if (!next.text) {
        setLogs((current) => [...current, next]);
        continue;
      }

      // The content is real backend stdout; only the reveal animation is cosmetic.
      const maxDelay = next.text.length > 120 ? 1 : next.text.length > 70 ? 2 : 4;
      for (let i = 1; i <= next.text.length; i += 1) {
        if (generation.current !== localGeneration) break;
        setTyping({ ...next, text: next.text.slice(0, i) });
        await sleep(maxDelay);
      }

      if (generation.current !== localGeneration) break;
      setLogs((current) => [...current, next]);
      setTyping(null);
    }

    logTyperBusy.current = false;
  };

  const enqueueLog = (entry: LogEntry) => {
    logQueue.current.push(entry);
    setPendingCount((count) => count + 1);
    void drainLogs(generation.current);
  };

  const resetRunUi = () => {
    generation.current += 1;
    logQueue.current = [];
    logTyperBusy.current = false;
    setLogs([]);
    setTyping(null);
    setPendingCount(0);
    setElapsed(0);
    setCurrentStep(0);
    setDone(false);
    setFailed(false);
    setBackendFinished(false);
  };

  const handleBackendEvent = (event: BackendEvent) => {
    if (event.type === "stage") {
      setCurrentStep(event.step);
      return;
    }

    if (event.type === "log") {
      enqueueLog({
        text: event.text,
        kind: event.kind ?? "info",
      });
      return;
    }

    if (event.type === "run_completed") {
      setBackendFinished(true);
      return;
    }

    if (event.type === "run_failed") {
      setFailed(true);
      setBackendFinished(true);
      if (event.message) enqueueLog({ text: `[ERROR] ${event.message}`, kind: "error" });
      return;
    }

    if (event.type === "run_cancelled") {
      setFailed(true);
      setBackendFinished(true);
    }
  };

  const runAudit = async () => {
    if (!backendOnline || running) return;

    resetRunUi();
    setRunning(true);

    unsubscribeRef.current?.();
    unsubscribeRef.current = null;

    try {
      const run = await startAudit();
      setRunId(run.id);

      unsubscribeRef.current = subscribeToRun(
        run.id,
        handleBackendEvent,
        () => {
          if (!backendFinished) {
            setFailed(true);
            setBackendFinished(true);
            enqueueLog({
              text: "[ERROR] Lost connection to the local reproducibility backend.",
              kind: "error",
            });
          }
        },
      );
    } catch (error) {
      setFailed(true);
      setRunning(false);
      enqueueLog({
        text: `[ERROR] ${error instanceof Error ? error.message : String(error)}`,
        kind: "error",
      });
    }
  };

  const stopAudit = async () => {
    if (!runId) return;
    try {
      await cancelAudit(runId);
    } catch (error) {
      enqueueLog({
        text: `[WARN] Could not cancel cleanly: ${
          error instanceof Error ? error.message : String(error)
        }`,
        kind: "warn",
      });
    }
  };

  useEffect(() => {
    return () => unsubscribeRef.current?.();
  }, []);

  const runStatus = useMemo(() => {
    if (!backendChecked) return "Checking backend";
    if (!backendOnline) return "Backend offline";
    if (failed) return "Audit stopped";
    if (done) return "All checks passed";
    if (running) return pipelineSteps[currentStep]?.title ?? "Running";
    return "Ready";
  }, [backendChecked, backendOnline, failed, done, running, currentStep]);

  return (
    <>
      <main className="app-shell">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <span /><span /><span />
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
            <div className={`backend-badge ${backendOnline ? "online" : "offline"}`}>
              <i />
              {backendOnline ? "Python backend online" : "Python backend offline"}
              <small>{API_BASE.replace(/^https?:\/\//, "")}</small>
            </div>

            <div className="run-state">
              <small>Current state</small>
              <strong>{runStatus}</strong>
            </div>

            {running ? (
              <button className="secondary-button danger" onClick={stopAudit}>
                <Icon name="stop" />
                Stop audit
              </button>
            ) : (
              <button
                className="primary-button"
                onClick={runAudit}
                disabled={!backendOnline}
                title={
                  backendOnline
                    ? "Run the real lightweight reproducibility audit"
                    : "Start the Python backend first"
                }
              >
                <Icon name="run" />
                Run real audit
              </button>
            )}
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
          failed={failed}
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
                    Orthogonal panels illustrating a beneficial LCC case and a canonical
                    postprocessing collapse.
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
            <LiveLog logs={logs} typing={typing} done={done} failed={failed} />
            <CodeMap />
          </div>
        </section>

        <footer>
          <span>
            Cross-cohort segmentation reliability · External validation · Failure attribution
          </span>
          <span>Public research demo</span>
        </footer>
      </main>

      <ImageModal item={selectedCase} onClose={() => setSelectedCase(null)} />
    </>
  );
}
