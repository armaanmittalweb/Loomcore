import { describe, expect, it } from 'vitest';
import { explain } from '../src/panels/BenchPanel';
import type { Bench } from '../src/types';

const bench = (fp: number, i8: number, cpu: Bench['cpu']): Bench => ({
  status: 'ready',
  rows: [
    { model: 'mobilenetv2', precision: 'FP32', mean: fp, p50: fp, p95: fp, n: 100 },
    { model: 'mobilenetv2', precision: 'INT8', mean: i8, p50: i8, p95: i8, n: 100 },
  ],
  cpu,
});
const n1 = { arch: 'aarch64', model: 'Ampere Altra (Neoverse N1)', logicalCpus: 4, flags: { asimd: true, asimddp: true, i8mm: false, bf16: false, sve: false } };

describe('explain (benchmark panel)', () => {
  it('on Arm, names SDOT/UDOT and states the measured direction, either way', () => {
    const faster = explain(bench(20, 16, n1));
    expect(faster).toMatch(/SDOT\/UDOT/);
    expect(faster).toMatch(/1\.25x as fast as FP32/);
    expect(faster).not.toMatch(/VNNI \(|AVX2 kernels/);
    const slower = explain(bench(20, 25, n1));
    expect(slower).toMatch(/0\.80x FP32, slower/);
    const same = explain(bench(20, 20, n1));
    expect(same).toMatch(/no real difference/);
  });
  it('on Arm with i8mm, names SMMLA; without asimddp, says plain NEON', () => {
    expect(explain(bench(10, 8, { ...n1, flags: { ...n1.flags, i8mm: true } }))).toMatch(/i8mm, SMMLA/);
    expect(explain(bench(10, 12, { ...n1, flags: { ...n1.flags, asimddp: false } }))).toMatch(/plain NEON/);
  });
  it('keeps the x86 wording', () => {
    const x86 = { arch: 'x86_64', model: 'AMD Ryzen 9 6900HX', logicalCpus: 16, flags: { avx2: true, avx512f: false, avx512_vnni: false, avx_vnni: false, amx_int8: false } };
    expect(explain(bench(8.172, 8.779, x86))).toMatch(/No VNNI on this CPU.*0\.93x FP32, slower/);
    const vnni = { ...x86, flags: { ...x86.flags, avx512f: true, avx512_vnni: true } };
    expect(explain(bench(10.6, 10.0, vnni))).toMatch(/This CPU has VNNI/);
  });
});
