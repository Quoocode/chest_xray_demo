export type RiskLevel = "High" | "Moderate" | "Low";

export type PredictionItem = {
  label: string;
  probability: number;
};

export type PredictionResponse = {
  filename: string;
  predictions: PredictionItem[];
  heatmap: string;
};

export type DisplayPredictionItem = PredictionItem & {
  percentage: number;
  risk: RiskLevel;
};
