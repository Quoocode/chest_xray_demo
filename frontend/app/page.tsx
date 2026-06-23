"use client";

import { useEffect, useMemo, useState } from "react";
import { PredictionResults } from "@/components/PredictionResults";
import { UploadCard } from "@/components/UploadCard";
import { ApiError, isSupportedImageFile, predictXray } from "@/lib/api";
import type { PredictionItem } from "@/lib/types";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionItem[]>([]);
  const [heatmapBase64, setHeatmapBase64] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Upload a JPG or PNG chest X-ray image.");

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  function handleFileSelect(file: File | null) {
    if (!file) {
      setSelectedFile(null);
      setPredictions([]);
      setHeatmapBase64(null);
      setErrorMessage(null);
      setStatusMessage("Upload a JPG or PNG chest X-ray image.");
      return;
    }

    if (file.size === 0) {
      setSelectedFile(null);
      setPredictions([]);
      setHeatmapBase64(null);
      setErrorMessage("The selected file is empty.");
      setStatusMessage("Upload a JPG or PNG chest X-ray image.");
      return;
    }

    if (!isSupportedImageFile(file)) {
      setSelectedFile(null);
      setPredictions([]);
      setHeatmapBase64(null);
      setErrorMessage("Only JPG and PNG files are supported.");
      setStatusMessage("Upload a JPG or PNG chest X-ray image.");
      return;
    }

    setSelectedFile(file);
    setPredictions([]);
    setHeatmapBase64(null);
    setErrorMessage(null);
    setStatusMessage(`Ready to analyze: ${file.name}`);
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setErrorMessage("Please choose a JPG or PNG image first.");
      return;
    }

    if (selectedFile.size === 0) {
      setErrorMessage("The selected file is empty.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setStatusMessage("Analyzing chest X-ray...");

    try {
      const response = await predictXray(selectedFile, 30000);
      setPredictions(response.predictions);
      setHeatmapBase64(response.heatmap);
      setStatusMessage(`Analysis complete for ${response.filename}`);
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Unexpected error while analyzing the image.");
      }
      setPredictions([]);
      setHeatmapBase64(null);
      setStatusMessage("Analysis failed.");
    } finally {
      setIsLoading(false);
    }
  }

  const summary = useMemo(() => {
    if (predictions.length === 0) {
      return "Upload a chest X-ray image to generate a rule-based summary.";
    }

    const topPrediction = [...predictions].sort((left, right) => right.probability - left.probability)[0];

    if (topPrediction?.label === "No Finding" && topPrediction.probability > 0.5) {
      return "The model suggests no major abnormal finding with moderate confidence.";
    }

    if (topPrediction) {
      return `The model highlights ${topPrediction.label} as the most probable finding. Clinical confirmation is required.`;
    }

    return "Upload a chest X-ray image to generate a rule-based summary.";
  }, [predictions]);

  const disclaimer = "This is a research demo and must not be used as a medical diagnosis.";

  function handleReset() {
    setSelectedFile(null);
    setPredictions([]);
    setHeatmapBase64(null);
    setErrorMessage(null);
    setIsLoading(false);
    setStatusMessage("Upload a JPG or PNG chest X-ray image.");
  }

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,#f8fbff_0%,#eef5ff_100%)]">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <header className="overflow-hidden rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm sm:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl space-y-4">
              <div className="inline-flex rounded-full border border-sky-200 bg-sky-50 px-4 py-2 text-sm font-semibold text-sky-700">
                Research Demo / Not for clinical use
              </div>
              <div className="space-y-3">
                <h1 className="text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
                  Chest X-ray AI Assistant
                </h1>
                <p className="text-base leading-7 text-slate-600 sm:text-lg">
                  Multi-label CheXpert disease screening demo
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:min-w-[360px] lg:grid-cols-1">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">Mode</p>
                <p className="mt-1 text-sm font-semibold text-slate-900">Medical demo dashboard</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">Endpoint</p>
                <p className="mt-1 text-sm font-semibold text-slate-900">POST /predict</p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">Status</p>
                <p className="mt-1 text-sm font-semibold text-slate-900">{statusMessage}</p>
              </div>
            </div>
          </div>

          {errorMessage ? (
            <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {errorMessage}
            </div>
          ) : null}
        </header>

        <section className="grid gap-8 xl:grid-cols-[420px_1fr]">
          <UploadCard
            file={selectedFile}
            previewUrl={previewUrl}
            isLoading={isLoading}
            errorMessage={errorMessage}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
          />

          <PredictionResults
            originalImageUrl={previewUrl}
            heatmapBase64={heatmapBase64}
            predictions={predictions}
            summary={summary}
            disclaimer={disclaimer}
          />
        </section>
      </div>
    </main>
  );
}
