"use client";

import { useRef, useState, type DragEvent } from "react";

type UploadCardProps = {
  file: File | null;
  previewUrl: string | null;
  isLoading: boolean;
  errorMessage: string | null;
  onFileSelect: (file: File | null) => void;
  onAnalyze: () => void;
  onReset: () => void;
};

export function UploadCard({
  file,
  previewUrl,
  isLoading,
  errorMessage,
  onFileSelect,
  onAnalyze,
  onReset,
}: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    onFileSelect(event.dataTransfer.files?.[0] ?? null);
  }

  function triggerPick() {
    inputRef.current?.click();
  }

  return (
    <section className="rounded-[2rem] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-sm">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-700">Upload Card</p>
          <h2 className="mt-2 text-xl font-semibold text-slate-900">Chest X-ray input</h2>
          <p className="mt-1 text-sm text-slate-500">Drag & drop or browse JPG/PNG files from your device.</p>
        </div>
        {file ? (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onReset();
            }}
            className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
          >
            Reset
          </button>
        ) : null}
      </div>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={triggerPick}
        className={`group relative flex min-h-[280px] cursor-pointer flex-col items-center justify-center rounded-[1.5rem] border-2 border-dashed px-6 py-8 text-center transition ${
          isDragging ? "border-sky-500 bg-sky-50" : "border-slate-200 bg-slate-50/70 hover:border-sky-300 hover:bg-sky-50/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png"
          className="hidden"
          onChange={(event) => onFileSelect(event.target.files?.[0] ?? null)}
        />

        {previewUrl ? (
          <div className="flex w-full max-w-md flex-col items-center gap-4">
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <img src={previewUrl} alt="Chest X-ray preview" className="max-h-72 w-full object-contain" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-900">{file?.name}</p>
              <p className="mt-1 text-xs text-slate-500">JPG / PNG preview ready for prediction</p>
            </div>
          </div>
        ) : (
          <div className="flex max-w-sm flex-col items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-sky-100 text-2xl text-sky-700 shadow-sm">
              +
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-900">Drop chest X-ray here</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Choose a JPG or PNG image to preview and analyze.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row">
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onAnalyze();
          }}
          disabled={isLoading || !file}
          className="inline-flex flex-1 items-center justify-center gap-3 rounded-2xl bg-sky-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-sky-300"
        >
          {isLoading ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Predicting...
            </>
          ) : (
            "Analyze"
          )}
        </button>
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onReset();
          }}
          disabled={!file && !previewUrl && !errorMessage}
          className="rounded-2xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
        >
          Reset
        </button>
      </div>

      {errorMessage ? (
        <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {errorMessage}
        </div>
      ) : null}
    </section>
  );
}
