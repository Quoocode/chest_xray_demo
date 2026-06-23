import type { PredictionResponse } from "@/lib/types";

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function isSupportedImageFile(file: File | null): boolean {
  if (!file) {
    return false;
  }

  const supportedMimeTypes = ["image/jpeg", "image/jpg", "image/png"];
  const supportedExtensions = /\.(jpe?g|png)$/i;

  return supportedMimeTypes.includes(file.type) || supportedExtensions.test(file.name);
}

export async function predictXray(file: File, timeoutMs = 30000): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const baseUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;

  try {
    response = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("Request timed out. Please try again.");
    }

    throw new ApiError("Unable to reach backend. Check whether the API is running.");
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    const detail = typeof errorBody.detail === "string" ? errorBody.detail : "Prediction failed";
    throw new ApiError(detail, response.status);
  }

  return response.json();
}
