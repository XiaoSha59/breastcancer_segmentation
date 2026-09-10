export type BackendEvent =
  | { type: "run_started"; timestamp: string; status: string }
  | { type: "stage"; timestamp: string; step: number; title: string }
  | {
      type: "log";
      timestamp: string;
      text: string;
      kind: "info" | "pass" | "warn" | "error";
    }
  | { type: "run_completed"; timestamp: string; status: string }
  | {
      type: "run_failed";
      timestamp: string;
      status: string;
      message?: string;
    }
  | { type: "run_cancelled"; timestamp: string; status: string };

export type HealthResponse = {
  ok: boolean;
  mode: string;
  audit_script: string;
  frontend_build_available: boolean;
  active_run: null | {
    id: string;
    status: string;
    current_step: number;
    current_step_title: string;
  };
};

export type RunResponse = {
  id: string;
  status: string;
  current_step: number;
  current_step_title: string;
  started_at: string | null;
  completed_at: string | null;
  return_code: number | null;
  event_count: number;
};

const configured = (
  import.meta.env.VITE_API_BASE_URL as string | undefined
)?.trim();

export const API_BASE =
  configured || "http://127.0.0.1:8768";

async function parseJson<T>(
  response: Response,
): Promise<T> {
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;

    try {
      const body = await response.json();
      message =
        body?.detail?.message ??
        body?.detail ??
        message;
    } catch {
      // Keep the HTTP status text.
    }

    throw new Error(String(message));
  }

  return response.json() as Promise<T>;
}

export async function checkBackend(): Promise<HealthResponse> {
  const response = await fetch(
    `${API_BASE}/api/health`,
  );

  return parseJson<HealthResponse>(response);
}

export async function startAudit(): Promise<RunResponse> {
  const response = await fetch(
    `${API_BASE}/api/runs`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );

  return parseJson<RunResponse>(response);
}

export async function cancelAudit(
  runId: string,
): Promise<RunResponse> {
  const response = await fetch(
    `${API_BASE}/api/runs/${runId}/cancel`,
    {
      method: "POST",
    },
  );

  return parseJson<RunResponse>(response);
}

export function subscribeToRun(
  runId: string,
  onEvent: (event: BackendEvent) => void,
  onConnectionError: () => void,
): () => void {
  const source = new EventSource(
    `${API_BASE}/api/runs/${runId}/events`,
  );

  let terminalEventReceived = false;

  source.onmessage = (message) => {
    const event = JSON.parse(
      message.data,
    ) as BackendEvent;

    if (
      event.type === "run_completed" ||
      event.type === "run_failed" ||
      event.type === "run_cancelled"
    ) {
      terminalEventReceived = true;
    }

    onEvent(event);

    // Closing the EventSource ourselves after a terminal event prevents the
    // browser from treating the server's normal end-of-stream as a failure.
    if (terminalEventReceived) {
      source.close();
    }
  };

  source.onerror = () => {
    source.close();

    if (!terminalEventReceived) {
      onConnectionError();
    }
  };

  return () => source.close();
}
