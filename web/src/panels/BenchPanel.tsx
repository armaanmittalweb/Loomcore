import type { Bench, BenchRow, CpuFlags } from '../types';
import { day, ms, ratio, shortCpu } from '../format';
import { Panel, type Origin, SourceTag } from '../ui';

// docs/BENCHMARKS.md, measured on the reference machine (no AVX-512, so no VNNI).
const REFERENCE = {
  machine: 'AMD Ryzen 9 6900HX',
  rows: [
    { model: 'mobilenetv2', precision: 'FP32', p50: 8.172 },
    { model: 'mobilenetv2', precision: 'INT8', p50: 8.779 },
    { model: 'bert_tiny', precision: 'FP32', p50: 0.278 },
    { model: 'bert_tiny', precision: 'INT8', p50: 0.247 },
  ],
};
const DOC = 'https://github.com/armaanmittalweb/loomcore/blob/main/docs/BENCHMARKS.md';
const MODELS = [
  { id: 'mobilenetv2', name: 'MobileNetV2', how: 'static QDQ' },
  { id: 'bert_tiny', name: 'bert_tiny', how: 'dynamic' },
];

const p50 = (rows: { model: string; precision: string; p50: number }[], model: string, prec: string) => rows.find((r) => r.model === model && r.precision === prec)?.p50;

function hasVnni(f: CpuFlags | null | undefined): boolean | null {
  if (!f) return null;
  if (f.avx512_vnni || f.avx_vnni || f.amx_int8) return true;
  if (f.avx512_vnni === null && f.avx_vnni === null) return null;
  return false;
}

const isArm = (bench: Bench) => bench.cpu?.arch === 'aarch64';

/** On Arm, ONNX Runtime's MLAS runs int8 GEMM and convolution on SDOT/UDOT
 * (asimddp), or SMMLA with i8mm. States what the measured ratio shows. */
function explainArm(bench: Bench, speed: number): string {
  const f = bench.cpu?.flags ?? {};
  const how = f.i8mm
    ? 'the int8 matrix-multiply instructions (i8mm, SMMLA) and the int8 dot product (SDOT/UDOT)'
    : f.asimddp
      ? 'the int8 dot-product instructions (SDOT/UDOT, the asimddp feature)'
      : 'plain NEON, since this CPU reports no int8 dot product (asimddp)';
  const outcome =
    speed >= 1.02
      ? `INT8 MobileNetV2 measures ${ratio(speed)} as fast as FP32: the int8 kernels outrun the Quantize/Dequantize nodes static QDQ quantization adds`
      : speed <= 0.98
        ? `INT8 MobileNetV2 measures ${ratio(speed)} FP32, slower: at batch size 1 on one thread the Quantize/Dequantize nodes static QDQ quantization adds cost more than the int8 kernels save`
        : `INT8 MobileNetV2 measures ${ratio(speed)} FP32, no real difference: the int8 kernels' saving and the cost of the Quantize/Dequantize nodes static QDQ quantization adds about cancel out at batch size 1 on one thread`;
  return `On this Arm CPU, ONNX Runtime's int8 GEMM and convolution kernels (MLAS) use ${how}. ${outcome}. The reference Ryzen, without VNNI, measured 0.93x. Whether INT8 pays is hardware-dependent, which is why precision is a runtime policy rather than a constant.`;
}

/** Why INT8 MobileNetV2 came out the way it did on this CPU, in one paragraph. */
export function explain(bench: Bench): string {
  const rows = bench.rows ?? [];
  const fp = p50(rows, 'mobilenetv2', 'FP32');
  const i8 = p50(rows, 'mobilenetv2', 'INT8');
  if (!fp || !i8) return '';
  const speed = fp / i8;
  if (isArm(bench)) return explainArm(bench, speed);
  const vnni = hasVnni(bench.cpu?.flags);
  const faster = speed >= 1.02;
  const slower = speed <= 0.98;
  if (vnni === true) {
    return faster
      ? `This CPU has VNNI (int8 dot-product instructions), so ONNX Runtime's INT8 convolutions use them and INT8 MobileNetV2 is ${ratio(speed)} as fast as FP32. The gain is modest: static QDQ quantization adds Quantize/Dequantize nodes, and at batch size 1 on one thread their cost eats into what VNNI saves.`
      : `This CPU has VNNI, yet INT8 MobileNetV2 is only ${ratio(speed)} FP32 here: the Quantize/Dequantize nodes that static QDQ quantization adds cost about as much as VNNI saves at batch size 1 on one thread.`;
  }
  if (vnni === false) {
    return slower
      ? `No VNNI on this CPU, so INT8 falls back to plain AVX2 kernels and the extra Quantize/Dequantize nodes cost more than they save: INT8 MobileNetV2 is ${ratio(speed)} FP32, slower, as on the reference Ryzen. That is the honest result, and why precision is a runtime policy rather than a constant.`
      : `No VNNI on this CPU, so INT8 runs on AVX2 kernels and comes out at ${ratio(speed)} FP32: no real gain, as on the reference Ryzen. That is why precision is a runtime policy rather than a constant.`;
  }
  return `INT8 MobileNetV2 runs at ${ratio(speed)} FP32 here. Whether INT8 pays depends on VNNI support, which this CPU did not report.`;
}

function Flag({ name, on }: { name: string; on: boolean | null | undefined }) {
  return (
    <span class={`flag ${on ? 'on' : on === false ? 'off' : 'unk'}`}>
      <i aria-hidden="true">{on ? '✓' : on === false ? '–' : '?'}</i>
      {name}
      <span class="sr-only">{on ? ' present' : on === false ? ' absent' : ' not reported'}</span>
    </span>
  );
}

function PairBars({ fp, i8 }: { fp?: number; i8?: number }) {
  const max = Math.max(fp ?? 0, i8 ?? 0) || 1;
  return (
    <span class="pair" aria-hidden="true">
      <i class="fp32" style={{ width: `${((fp ?? 0) / max) * 100}%` }} />
      <i class="int8" style={{ width: `${((i8 ?? 0) / max) * 100}%` }} />
    </span>
  );
}

export function BenchPanel({ bench, origin }: { bench: Bench; origin: Origin }) {
  const rows: BenchRow[] = bench.rows ?? [];
  const cpu = bench.cpu;
  const where = origin.kind === 'live' ? 'The live runtime' : 'Recorded machine';
  return (
    <Panel id="bench" kicker="Benchmark" title="FP32 vs INT8, measured" tag={<SourceTag origin={origin} />} class="p-bench">
      {bench.status !== 'ready' ? (
        <p class="fine">{bench.status === 'running' || bench.status === 'pending' ? 'The benchmark runs once when the runtime starts; it is running now.' : `No benchmark on this machine: ${bench.reason ?? bench.status}.`}</p>
      ) : (
        <>
          <p class="bench-cpu">
            <span class="mono">{cpu ? shortCpu(cpu.model) : 'unknown CPU'}</span>
            {isArm(bench) ? (
              <span class="flags-row">
                <Flag name="NEON" on={cpu?.flags?.asimd} />
                <Flag name="dot product (SDOT)" on={cpu?.flags?.asimddp} />
                <Flag name="i8mm" on={cpu?.flags?.i8mm} />
                <Flag name="BF16" on={cpu?.flags?.bf16} />
                <Flag name="SVE" on={cpu?.flags?.sve} />
              </span>
            ) : (
              <span class="flags-row">
                <Flag name="AVX2" on={cpu?.flags?.avx2} />
                <Flag name="AVX-512" on={cpu?.flags?.avx512f} />
                <Flag name="AVX-512 VNNI" on={cpu?.flags?.avx512_vnni} />
                <Flag name="AVX-VNNI" on={cpu?.flags?.avx_vnni} />
              </span>
            )}
          </p>
          <table class="tbl bench-tbl">
            <caption class="sr-only">p50 latency per inference, milliseconds, single item, one thread per session</caption>
            <thead>
              <tr>
                <th scope="col">model</th>
                <th scope="col" class="num">FP32 p50</th>
                <th scope="col" class="num">INT8 p50</th>
                <th scope="col" class="bench-pair-h">
                  <span class="lg">
                    <i class="sw sw-fp32" aria-hidden="true" />FP32 <i class="sw sw-int8" aria-hidden="true" />INT8
                  </span>
                </th>
                <th scope="col" class="num">INT8 speed-up</th>
                <th scope="col" class="num">on {REFERENCE.machine}</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => {
                const fp = p50(rows, m.id, 'FP32');
                const i8 = p50(rows, m.id, 'INT8');
                const rfp = p50(REFERENCE.rows, m.id, 'FP32') as number;
                const ri8 = p50(REFERENCE.rows, m.id, 'INT8') as number;
                return (
                  <tr key={m.id}>
                    <th scope="row">
                      {m.name}
                      <span class="muted"> · {m.how}</span>
                    </th>
                    <td class="num">{ms(fp)}</td>
                    <td class="num">{ms(i8)}</td>
                    <td>
                      <PairBars fp={fp} i8={i8} />
                    </td>
                    <td class="num strong">{fp && i8 ? ratio(fp / i8) : '-'}</td>
                    <td class="num">{ratio(rfp / ri8)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p class="bench-why">{explain(bench)}</p>
          <p class="fine">
            {where}: <code>loomcore_bench</code>, {bench.warmup ?? 20} warm-up and {bench.iters ?? 100} measured runs per variant
            {bench.ranAt ? `, ${day(bench.ranAt)}` : ''}; ONNX Runtime 1.30, one intra-op thread per session (the scheduler supplies the parallelism). Latency only: the INT8 models are not checked for accuracy. Reference numbers from <a href={DOC}>docs/BENCHMARKS.md</a>.
          </p>
        </>
      )}
    </Panel>
  );
}
