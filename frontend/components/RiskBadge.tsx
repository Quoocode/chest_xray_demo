import type { RiskLevel } from "@/lib/types";

type RiskBadgeProps = {
  risk: RiskLevel;
};

export function RiskBadge({ risk }: RiskBadgeProps) {
  const className =
    risk === "High"
      ? "border-red-200 bg-red-50 text-red-700"
      : risk === "Moderate"
        ? "border-amber-200 bg-amber-50 text-amber-700"
        : "border-emerald-200 bg-emerald-50 text-emerald-700";

  return <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${className}`}>{risk}</span>;
}
