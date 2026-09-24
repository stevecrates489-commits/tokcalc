"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle2, Upload, Beaker, Download } from "lucide-react";
import {
  vllmParser,
} from "@/lib/benchmark-parser-vllm";
import { sglangParser } from "@/lib/benchmark-parser-sglang";
import { trtllmParser } from "@/lib/benchmark-parser-trtllm";
import { tokcalcStandardParser, BENCHMARK_TEMPLATE } from "@/lib/benchmark-parser-tokcalc";
import type {
  BenchmarkRecord,
  BenchmarkParser,
  CalibrationResult,
} from "@/lib/benchmark-schema";
import {
  formatCalibrationVerdict,
  generateBenchmarkId,
} from "@/lib/benchmark-schema";
import type { CalcResult } from "@/lib/token-calc";
import { fmtTokens, fmtMs } from "@/lib/token-calc";
import { track } from "@/lib/track";

interface BenchmarkImportProps {
  /** The theoretical estimate from tokcalc's formulas */
  estimate: CalcResult;
  /** Current model name (for matching) */
  modelName: string;
  /** Current GPU name (for matching) */
  gpuName: string;
}

export function BenchmarkImport({ estimate, modelName, gpuName }: BenchmarkImportProps) {
  const [rawText, setRawText] = useState("");
  const [record, setRecord] = useState<BenchmarkRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const handleParse = () => {
    setError(null);
    if (!rawText.trim()) {
      setError("Paste a benchmark JSON output first.");
      return;
    }

    // Try parsers — order matters: most-specific first, most-generic last.
    // TRT-LLM has unique field names (first_token_latency_avg, inter_token_latency_avg)
    // SGLang has unique field names (total_throughput, ttft_avg, itl_avg)
    // vLLM is most generic (output_throughput, mean_ttft_ms) — try last
    const parsers: BenchmarkParser[] = [trtllmParser, sglangParser, vllmParser, tokcalcStandardParser];
    for (const parser of parsers) {
      if (parser.detect(rawText)) {
        const parsed = parser.parse(rawText);
        if (parsed) {
          setRecord(parsed);
          track("imported_benchmark", {
            engine: parsed.serving_engine,
            model: parsed.model_display_name,
            throughput: parsed.output_token_throughput_tps,
            parser: parser.name,
          });
          setExpanded(true);
          return;
        }
      }
    }

    setError(
      "Couldn't parse this as vLLM, SGLang, or TensorRT-LLM benchmark JSON. " +
      "Make sure you're pasting the raw JSON output from the benchmark script. " +
      "Supported formats: vLLM benchmark_serving.py, SGLang benchmark, TRT-LLM benchmark_serving.py."
    );
  };

  // Compute calibration if we have both estimate and observed data
  const calibration: CalibrationResult | null = record
    ? computeCalibration(estimate, record)
    : null;

  const verdict = calibration ? formatCalibrationVerdict(calibration.verdict) : null;

  return (
    <Card className="border-border/60 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Beaker className="size-4 text-emerald-500" />
          Observed benchmark — calibrate the formula
        </CardTitle>
        <CardDescription className="text-xs">
          Paste a vLLM benchmark JSON output to see how tokcalc&apos;s theoretical estimates compare to real-world measured performance.
          The calibration verdict shows whether the formula is accurate, too conservative, or too optimistic.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {!record && !expanded && (
          <div className="flex flex-col sm:flex-row gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setExpanded(true)}
              className="flex-1 gap-1.5"
            >
              <Upload className="size-3.5" />
              Import a benchmark result
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                const blob = new Blob([BENCHMARK_TEMPLATE], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "tokcalc-benchmark-template.json";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                track("downloaded_benchmark_template");
              }}
              className="flex-1 gap-1.5"
            >
              <Download className="size-3.5" />
              Download template
            </Button>
          </div>
        )}

        {expanded && !record && (
          <>
            <div>
              <Textarea
                placeholder={`Paste JSON from:\n- vllm benchmark_serving.py output\n- SGLang benchmark output\n- TensorRT-LLM benchmark_serving.py output\n- tokcalc community template (download above)\n\nOr click "Download template" to get a structured JSON you can fill in manually.`}
                value={rawText}
                onChange={(e) => setRawText(e.target.value)}
                className="font-mono text-xs min-h-[160px] resize-y"
              />
            </div>
            {error && (
              <div className="text-[11px] text-amber-600 dark:text-amber-400 flex items-start gap-1.5">
                <AlertTriangle className="size-3.5 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
            <div className="flex gap-2">
              <Button size="sm" onClick={handleParse} className="gap-1.5">
                <Beaker className="size-3.5" />
                Parse &amp; calibrate
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => { setExpanded(false); setRawText(""); setError(null); }}
              >
                Cancel
              </Button>
            </div>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Your benchmark data stays in your browser — nothing is sent to a server.
              SGLang and TensorRT-LLM parsers are on the roadmap.
            </p>
          </>
        )}

        {record && calibration && verdict && (
          <div className="space-y-3">
            {/* Calibration verdict */}
            <div className={`flex items-start gap-3 p-3 rounded-md border ${
              calibration.verdict === "validated"
                ? "border-emerald-500/40 bg-emerald-500/5"
                : calibration.verdict === "insufficient_data"
                  ? "border-border/60 bg-muted/20"
                  : "border-amber-500/40 bg-amber-500/5"
            }`}>
              {calibration.verdict === "validated" ? (
                <CheckCircle2 className="size-4 text-emerald-500 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="size-4 text-amber-500 shrink-0 mt-0.5" />
              )}
              <div className="text-xs">
                <div className={`font-medium ${verdict.color}`}>
                  {verdict.label}
                </div>
                <div className="text-muted-foreground mt-0.5 leading-relaxed">
                  {verdict.description}
                </div>
                {calibration.calibrationErrors.throughputErrorPct !== undefined && (
                  <div className="mt-1.5 font-mono text-[11px]">
                    Throughput: estimated <strong className="text-foreground">{fmtTokens(estimate.decodeTokensPerSec)} tok/s</strong> vs observed <strong className="text-foreground">{fmtTokens(record.output_token_throughput_tps)} tok/s</strong>
                    {" "}→ <span className={Math.abs(calibration.calibrationErrors.throughputErrorPct) < 20 ? "text-emerald-500" : "text-amber-500"}>
                      {calibration.calibrationErrors.throughputErrorPct > 0 ? "+" : ""}{calibration.calibrationErrors.throughputErrorPct.toFixed(1)}% error
                    </span>
                  </div>
                )}
                {calibration.calibrationErrors.ttftErrorPct !== undefined && (
                  <div className="mt-1 font-mono text-[11px]">
                    TTFT: estimated <strong className="text-foreground">{fmtMs(estimate.ttftMs)}</strong> vs observed <strong className="text-foreground">{fmtMs(record.ttft_mean_ms || 0)}</strong>
                    {" "}→ <span className={Math.abs(calibration.calibrationErrors.ttftErrorPct) < 20 ? "text-emerald-500" : "text-amber-500"}>
                      {calibration.calibrationErrors.ttftErrorPct > 0 ? "+" : ""}{calibration.calibrationErrors.ttftErrorPct.toFixed(1)}% error
                    </span>
                  </div>
                )}
                {calibration.calibrationErrors.itlErrorPct !== undefined && (
                  <div className="mt-1 font-mono text-[11px]">
                    ITL: estimated <strong className="text-foreground">{fmtMs(estimate.itlMs)}</strong> vs observed <strong className="text-foreground">{fmtMs(record.itl_mean_ms || 0)}</strong>
                    {" "}→ <span className={Math.abs(calibration.calibrationErrors.itlErrorPct) < 20 ? "text-emerald-500" : "text-amber-500"}>
                      {calibration.calibrationErrors.itlErrorPct > 0 ? "+" : ""}{calibration.calibrationErrors.itlErrorPct.toFixed(1)}% error
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Observed metrics detail */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <div className="space-y-0.5">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Observed throughput</div>
                <div className="font-mono text-sm font-medium">{fmtTokens(record.output_token_throughput_tps)} tok/s</div>
                <div className="text-[10px] text-muted-foreground">{record.request_throughput_rps?.toFixed(1) || "—"} req/s</div>
              </div>
              <div className="space-y-0.5">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">TTFT (mean / p99)</div>
                <div className="font-mono text-sm font-medium">{fmtMs(record.ttft_mean_ms || 0)}</div>
                <div className="text-[10px] text-muted-foreground">p99: {fmtMs(record.ttft_p99_ms || 0)}</div>
              </div>
              <div className="space-y-0.5">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">ITL (mean / p99)</div>
                <div className="font-mono text-sm font-medium">{fmtMs(record.itl_mean_ms || 0)}</div>
                <div className="text-[10px] text-muted-foreground">p99: {fmtMs(record.itl_p99_ms || 0)}</div>
              </div>
              <div className="space-y-0.5">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Config</div>
                <div className="font-mono text-[11px] font-medium">{record.gpu_count}×{record.accelerator_model.split(" ")[0]}</div>
                <div className="text-[10px] text-muted-foreground">{record.serving_engine} v{record.engine_version.slice(0, 6)}</div>
              </div>
            </div>

            {/* Provenance */}
            <div className="text-[10px] text-muted-foreground border-t border-border/40 pt-2">
              <strong className="text-foreground">Source:</strong> {record.source_name} ({record.confidence_tier.replace(/_/g, " ")}) ·
              {" "}{record.request_count.toLocaleString()} requests · {record.input_tokens_mean} in / {record.output_tokens_mean} out tokens ·
              {" "}concurrency {record.concurrency} ·
              {" "}{record.model_display_name} · {record.quantization_format} ·
              {" "}<a href={record.source_url} target="_blank" rel="noopener noreferrer" className="text-emerald-500 hover:underline">source ↗</a>
            </div>

            <Button
              size="sm"
              variant="ghost"
              onClick={() => { setRecord(null); setRawText(""); setExpanded(false); }}
              className="text-xs"
            >
              Import a different benchmark
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ---------- Calibration computation ---------- */

function computeCalibration(
  estimate: CalcResult,
  observed: BenchmarkRecord,
): CalibrationResult {
  const errors: CalibrationResult["calibrationErrors"] = {};

  // Throughput calibration (if observed has throughput)
  if (observed.output_token_throughput_tps && estimate.decodeTokensPerSec > 0) {
    errors.throughputErrorPct =
      ((observed.output_token_throughput_tps - estimate.decodeTokensPerSec) / estimate.decodeTokensPerSec) * 100;
  }

  // TTFT calibration (if observed has mean TTFT)
  if (observed.ttft_mean_ms && estimate.ttftMs > 0) {
    errors.ttftErrorPct =
      ((observed.ttft_mean_ms - estimate.ttftMs) / estimate.ttftMs) * 100;
  }

  // ITL calibration (if observed has mean ITL)
  if (observed.itl_mean_ms && estimate.itlMs > 0) {
    errors.itlErrorPct =
      ((observed.itl_mean_ms - estimate.itlMs) / estimate.itlMs) * 100;
  }

  // Memory calibration (if observed has peak GPU memory)
  if (observed.gpu_memory_peak_gb && estimate.totalVramNeededGb > 0) {
    errors.memoryErrorPct =
      ((observed.gpu_memory_peak_gb - estimate.totalVramNeededGb) / estimate.totalVramNeededGb) * 100;
  }

  // Determine verdict
  let verdict: CalibrationResult["verdict"] = "insufficient_data";
  const errorValues = Object.values(errors).filter((v) => v !== undefined) as number[];

  if (errorValues.length >= 2) {
    const maxAbsError = Math.max(...errorValues.map((v) => Math.abs(v)));
    if (maxAbsError < 20) {
      verdict = "validated";
    } else if (maxAbsError > 0) {
      // If most errors are positive (observed > estimated), we underestimated
      // If most are negative (observed < estimated), we overestimated
      const avgError = errorValues.reduce((a, b) => a + b, 0) / errorValues.length;
      verdict = avgError > 0 ? "underestimated" : "overestimated";
    }
  }

  return {
    estimated: {
      decodeTokensPerSec: estimate.decodeTokensPerSec,
      aggregateTokensPerSec: estimate.aggregateTokensPerSec,
      prefillTokensPerSec: estimate.prefillTokensPerSec,
      ttftMs: estimate.ttftMs,
      itlMs: estimate.itlMs,
      modelSizeGb: estimate.modelSizeGb,
    },
    observed: {
      outputTokenThroughputTps: observed.output_token_throughput_tps,
      ttftMeanMs: observed.ttft_mean_ms,
      ttftP95Ms: observed.ttft_p95_ms,
      itlMeanMs: observed.itl_mean_ms,
      itlP95Ms: observed.itl_p95_ms,
      gpuMemoryPeakGb: observed.gpu_memory_peak_gb,
    },
    calibrationErrors: errors,
    verdict,
  };
}
