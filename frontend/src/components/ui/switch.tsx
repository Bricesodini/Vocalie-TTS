import * as React from "react";

import { cn } from "@/lib/utils";

export type SwitchProps = {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
};

const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(
  ({ checked = false, onCheckedChange, disabled = false, className }, ref) => (
    <button
      ref={ref}
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onCheckedChange?.(!checked)}
      className={cn(
        "relative inline-flex h-6 w-11 items-center rounded-full border border-zinc-300 transition",
        checked ? "bg-zinc-900" : "bg-zinc-200",
        disabled && "cursor-not-allowed opacity-60",
        className
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 rounded-full bg-white shadow transition",
          checked ? "translate-x-5" : "translate-x-1"
        )}
      />
    </button>
  )
);
Switch.displayName = "Switch";

export { Switch };
