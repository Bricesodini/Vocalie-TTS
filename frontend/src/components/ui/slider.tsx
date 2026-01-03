import * as React from "react";

import { cn } from "@/lib/utils";

export type SliderProps = {
  value?: number[];
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  onValueChange?: (value: number[]) => void;
};

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ value = [0], min = 0, max = 100, step = 1, disabled = false, className, onValueChange }, ref) => {
    const current = Array.isArray(value) ? value[0] ?? min : min;
    return (
      <input
        ref={ref}
        type="range"
        min={min}
        max={max}
        step={step}
        value={current}
        disabled={disabled}
        onChange={(event) => onValueChange?.([Number(event.target.value)])}
        className={cn(
          "h-2 w-full appearance-none rounded-full bg-zinc-200",
          "accent-zinc-900 disabled:cursor-not-allowed",
          className
        )}
      />
    );
  }
);
Slider.displayName = "Slider";

export { Slider };
