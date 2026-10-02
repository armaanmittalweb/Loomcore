// The Space's API shapes (space/server.py). Times are milliseconds unless a
// name says otherwise; trace timestamps are microseconds, as the Chrome Trace
// Event Format has them.

export type Precision = 'FP32' | 'INT8';
export type Lane = 'CPU' | 'GPU_SIM';
export type PolicyName =
  | 'circuit-breaker'
  | 'bulkhead'
  | 'confidence-gate'
  | 'precision-planner'
  | 'latency-budget'
  | 'load-aware';

export interface TraceEvent {
  name: string;
  ph: string;
  ts?: number;
  dur?: number;
  pid?: number;
  tid?: number;
  id?: number;
  bp?: string;
  cat?: string;
  s?: string;
  args?: Record<string, unknown>;
}

export interface PolicyOpinion {
  policy: string;
  reason: string;
}

export interface Decision {
  node: string;
  policies: PolicyOpinion[];
  message: string;
}

export type RunStatus = 'ok' | 'rejected' | 'cancelled' | 'shed' | 'failed';

export interface RunResult {
  status: RunStatus;
  error: string | null;
  source: string;
  timeBudgetMs: number | null;
  policies: PolicyName[];
  graphSwapMs: number | null;
  jobId: string | null;
  label: string | null;
  confidence: number | null;
  top5: { label: string; p: number }[];
  text: string | null;
  embedding: number[] | null;
  skipped: string[];
  decisions: Decision[];
  nodes: { id: string; precision: Precision; backend: Lane; ms: number }[];
  totalMs: number;
  trace: TraceEvent[];
}

export interface NodeStats {
  id: string;
  count: number;
  p50: number;
  p95: number;
  mean: number;
  precisions: Partial<Record<Precision, number>>;
  lanes: Partial<Record<Lane, number>>;
}

export interface BatchStats {
  node: string;
  count: number;
  sizes: Record<string, number>;
  meanSize: number;
}

export interface LoadResult {
  jobs: number;
  concurrency: number;
  timeBudgetMs: number | null;
  policies: PolicyName[];
  source: string;
  graphSwapMs: number | null;
  submitted: number;
  completed: number;
  rejected: number;
  cancelled: number;
  shed: number;
  failed: number;
  wallMs: number;
  throughput: number;
  jobP50: number | null;
  jobP95: number | null;
  nodes: NodeStats[];
  batches: BatchStats[];
  decisions: { policy: string; count: number }[];
  skipped: Record<string, number>;
  trace: TraceEvent[];
}

export interface SwapTiming {
  index: number;
  buildMs: number;
  completedBefore: number;
  inFlight: number;
  atMs: number;
}

export interface ReloadResult {
  jobs: number;
  swaps: number;
  policies: PolicyName[];
  submitted: number;
  completed: number;
  lost: number;
  wallMs: number;
  swapTimings: SwapTiming[];
  nodes: NodeStats[];
  batches: BatchStats[];
  trace: TraceEvent[];
}

export interface Health {
  ok: boolean;
  ready: boolean;
  version: string;
  models: Record<string, Precision[]>;
  cpu: string;
  uptimeS: number;
  busy: boolean;
  bench: string;
}

export interface PolicyInfo {
  name: PolicyName;
  cpp: string;
  reads: string;
  decides: string;
}

export interface GraphNode {
  id: string;
  backend: Lane;
  priority: number;
  maxBatchSize: number;
  batchWindowMs: number;
  dependsOn: string[];
  variants: Precision[];
  confidenceSource: string | null;
  qualityWeight: number;
}

export interface GraphInfo {
  nodes: GraphNode[];
  edges: { from: string; to: string }[];
  topoOrder: string[];
  sinks: string[];
  policies: PolicyInfo[];
  activePolicies: PolicyName[];
  scheduler: {
    admissionControl: boolean;
    precisionPlanning: boolean;
    deadlineCancellation: boolean;
    edfScoring: boolean;
    cpuThreads: number;
    gpuSimThreads: number;
    gpuSimOverheadMs: number;
  };
  samples: { id: string; caption: string; author: string; licence: string; source: string }[];
}

/** x86: AVX2 / AVX-512 / VNNI / AMX. aarch64: NEON (asimd), the int8 dot
 * product (asimddp: SDOT/UDOT), int8 matrix multiply (i8mm), bf16, SVE. */
export interface CpuFlags {
  avx2?: boolean | null;
  avx512f?: boolean | null;
  avx512_vnni?: boolean | null;
  avx_vnni?: boolean | null;
  amx_int8?: boolean | null;
  asimd?: boolean | null;
  asimddp?: boolean | null;
  i8mm?: boolean | null;
  bf16?: boolean | null;
  sve?: boolean | null;
}

export interface BenchRow {
  model: string;
  precision: Precision;
  mean: number;
  p50: number;
  p95: number;
  n: number;
}

export interface Bench {
  status: 'pending' | 'running' | 'ready' | 'failed' | 'unavailable' | 'skipped';
  ranAt?: string;
  rows?: BenchRow[];
  cpu?: { arch?: string; model: string; logicalCpus: number; flags: CpuFlags | null };
  reason?: string;
  warmup?: number;
  iters?: number;
}

export interface RecordingMeta {
  recordedAt: string;
  source: 'space' | 'local';
  machine: string;
  os?: string;
  version: string;
}
