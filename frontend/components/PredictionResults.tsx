"use client";

import { useMemo, useState } from "react";
import { RiskBadge } from "@/components/RiskBadge";
import type { DisplayPredictionItem, PredictionItem } from "@/lib/types";

type PredictionResultsProps = {
  originalImageUrl: string | null;
  heatmapBase64: string | null;
  predictions: PredictionItem[];
  summary: string;
  disclaimer: string;
};

function getRisk(probability: number) {
  if (probability >= 0.7) return "High";
  if (probability >= 0.4) return "Moderate";
  return "Low";
}

function toDisplayItems(predictions: PredictionItem[]): DisplayPredictionItem[] {
  return [...predictions]
    .sort((left, right) => right.probability - left.probability)
    .map((item) => ({
      ...item,
      percentage: item.probability * 100,
      risk: getRisk(item.probability),
    }));
}

function ProgressBar({ probability }: { probability: number }) {
  const percentage = Math.max(0, Math.min(100, probability * 100));

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
      <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-blue-600" style={{ width: `${percentage}%` }} />
    </div>
  );
}

export function PredictionResults({
  originalImageUrl,
  heatmapBase64,
  predictions,
  summary,
  disclaimer,
}: PredictionResultsProps) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(65);

  const heatmapUrl = useMemo(() => {
    if (!heatmapBase64) {
      return null;
    }

    return `data:image/png;base64,${heatmapBase64}`;
  }, [heatmapBase64]);

  if (predictions.length === 0) {
    return (
      <section className="grid gap-6">
        <div className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">Result Dashboard</p>
          <h2 className="mt-2 text-xl font-semibold text-slate-900">No prediction yet</h2>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            Upload a chest X-ray image and run Analyze to view model results.
          </p>
        </div>
        <div className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 text-sm leading-6 text-slate-500 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          This is a research demo and must not be used as a medical diagnosis.
        </div>
      </section>
    );
  }

  const displayItems = toDisplayItems(predictions);
  const topFive = displayItems.slice(0, 5);

  return (
    <section className="grid gap-6">
      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <article className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">Original Xray</p>
          <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
            {originalImageUrl ? (
              <img src={originalImageUrl} alt="Original chest X-ray" className="h-auto w-full object-contain" />
            ) : (
              <div className="flex min-h-[320px] items-center justify-center text-sm text-slate-500">
                Select an image to preview it here.
              </div>
            )}
          </div>
        </article>

        <article className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">Grad-CAM Overlay</p>
              <h3 className="mt-2 text-lg font-semibold text-slate-900">Attention heatmap</h3>
            </div>
            <button
              type="button"
              onClick={() => setShowHeatmap((value) => !value)}
              className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
            >
              {showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
            </button>
          </div>

          <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
            {originalImageUrl ? (
              <div className="relative w-full">
                <img src={originalImageUrl} alt="Original chest X-ray for overlay" className="h-auto w-full object-contain" />
                {showHeatmap && heatmapUrl ? (
                  <img
                    src={heatmapUrl}
                    alt="Grad-CAM heatmap overlay"
                    className="absolute inset-0 h-full w-full object-contain"
                    style={{ opacity: heatmapOpacity / 100 }}
                  />
                ) : null}
              </div>
            ) : (
              <div className="flex min-h-[320px] items-center justify-center text-sm text-slate-500">
                Grad-CAM overlay will appear after analysis.
              </div>
            )}
          </div>

          <div className="mt-5 space-y-3">
            <div className="flex items-center justify-between text-sm text-slate-600">
              <span>Heatmap Opacity</span>
              <span className="font-semibold text-slate-900">{heatmapOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={heatmapOpacity}
              onChange={(event) => setHeatmapOpacity(Number(event.target.value))}
              className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-sky-600"
            />
          </div>
        </article>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <article className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">Top Findings</p>
          <div className="mt-4 space-y-4">
            {topFive.map((item, index) => (
              <div key={item.label} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">#{index + 1}</p>
                    <h3 className="mt-1 text-base font-semibold text-slate-900">{item.label}</h3>
                  </div>
                  <RiskBadge risk={item.risk} />
                </div>
                <div className="mt-4 flex items-center justify-between text-sm text-slate-600">
                  <span>Probability</span>
                  <span className="font-semibold text-slate-900">{item.percentage.toFixed(2)}%</span>
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-[2rem] border border-slate-200 bg-gradient-to-br from-sky-50 to-white p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">AI Summary</p>
          <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-5 text-sm leading-7 text-slate-700 shadow-sm">
            <p>{summary}</p>
            <p className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-amber-800">
              {disclaimer}
            </p>
          </div>
        </article>
      </div>

      <article className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">Full Prediction Table</p>
            <h2 className="mt-2 text-xl font-semibold text-slate-900">14-label CheXpert output</h2>
          </div>
          <p className="text-sm text-slate-500">Sorted in descending probability order.</p>
        </div>

        <div className="mt-6 overflow-hidden rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
            <thead className="bg-slate-50 text-slate-500">
              <tr>
                <th className="px-4 py-3 font-medium">Label</th>
                <th className="px-4 py-3 font-medium">Probability</th>
                <th className="px-4 py-3 font-medium">Risk</th>
                <th className="px-4 py-3 font-medium">Progress</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {displayItems.map((item, index) => (
                <tr key={item.label} className="align-top">
                  <td className="px-4 py-4">
                    <p className="font-medium text-slate-900">{index + 1}. {item.label}</p>
                  </td>
                  <td className="px-4 py-4 text-slate-700">{item.percentage.toFixed(2)}%</td>
                  <td className="px-4 py-4">
                    <RiskBadge risk={item.risk} />
                  </td>
                  <td className="px-4 py-4">
                    <div className="space-y-2">
                      <ProgressBar probability={item.probability} />
                      <p className="text-xs text-slate-400">{Math.round(item.percentage)}% of the bar</p>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <div className="rounded-[2rem] border border-slate-200 bg-white/95 p-5 text-sm leading-6 text-slate-500 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
        This is a research demo and must not be used as a medical diagnosis.
      </div>
    </section>
  );
}
