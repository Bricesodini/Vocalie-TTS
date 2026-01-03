import { Fragment } from "react";

import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { EngineSchemaField } from "@/lib/types";

type DynamicFieldsProps = {
  fields: EngineSchemaField[];
  values: Record<string, unknown>;
  context?: Record<string, unknown>;
  onChange: (key: string, value: unknown, field: EngineSchemaField) => void;
};

function isVisible(field: EngineSchemaField, context: Record<string, unknown>) {
  const rule = field.visible_if;
  if (!rule) return true;
  for (const [key, expected] of Object.entries(rule)) {
    if (key === "voice_count_min") {
      const count = Number(context.voice_count ?? 0);
      if (count < Number(expected)) return false;
      continue;
    }
    if (context[key] !== expected) return false;
  }
  return true;
}

function normalizeChoices(choices: Array<unknown> | undefined) {
  if (!choices) return [];
  return choices.map((choice) => {
    if (Array.isArray(choice) && choice.length === 2) {
      return { label: String(choice[0]), value: String(choice[1]) };
    }
    return { label: String(choice), value: String(choice) };
  });
}

export function DynamicFields({ fields, values, context = {}, onChange }: DynamicFieldsProps) {
  const mergedContext = { ...values, ...context };

  return (
    <div className="grid gap-4">
      {fields.map((field) => {
        if (!isVisible(field, mergedContext)) return null;
        const value = values[field.key];
        const label = field.label || field.key;
        if (field.type === "bool") {
          return (
            <label key={field.key} className="flex items-center justify-between rounded-md border border-zinc-200 px-3 py-2">
              <div>
                <p className="text-sm font-medium text-zinc-900">{label}</p>
                {field.help && <p className="text-xs text-zinc-500">{field.help}</p>}
              </div>
              <Switch checked={Boolean(value)} onCheckedChange={(checked) => onChange(field.key, checked, field)} />
            </label>
          );
        }

        if (field.type === "choice" || field.type === "select") {
          const choices = normalizeChoices(field.choices);
          return (
            <div key={field.key} className="grid gap-2">
              <label className="text-sm font-medium text-zinc-900">{label}</label>
              <Select
                value={value ? String(value) : ""}
                onValueChange={(next) => onChange(field.key, next, field)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Choisir" />
                </SelectTrigger>
                <SelectContent>
                  {choices.map((choice) => (
                    <SelectItem key={choice.value} value={choice.value}>
                      {choice.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {field.help && <p className="text-xs text-zinc-500">{field.help}</p>}
            </div>
          );
        }

        if (field.type === "float" || field.type === "int" || field.type === "slider") {
          const numeric = typeof value === "number" ? value : Number(field.default ?? 0);
          const min = field.min ?? 0;
          const max = field.max ?? 1;
          const step = field.step ?? 0.1;
          return (
            <Fragment key={field.key}>
              <div className="grid gap-2 rounded-md border border-zinc-200 px-3 py-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-zinc-900">{label}</span>
                  <span className="font-mono text-xs text-zinc-500">{numeric.toFixed(field.type === "int" ? 0 : 2)}</span>
                </div>
                <Slider
                  value={[numeric]}
                  min={min}
                  max={max}
                  step={step}
                  onValueChange={(vals) => onChange(field.key, field.type === "int" ? Math.round(vals[0]) : vals[0], field)}
                />
                {field.help && <p className="text-xs text-zinc-500">{field.help}</p>}
              </div>
            </Fragment>
          );
        }

        if (field.type === "str") {
          return (
            <div key={field.key} className="grid gap-2">
              <label className="text-sm font-medium text-zinc-900">{label}</label>
              <Input
                value={value ? String(value) : ""}
                onChange={(event) => onChange(field.key, event.target.value, field)}
                placeholder={field.help || ""}
              />
            </div>
          );
        }

        return null;
      })}
    </div>
  );
}
