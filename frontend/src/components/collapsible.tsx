"use client";

import { useState } from "react";

type CollapsibleProps = {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  badge?: string;
  className?: string;
};

export function Collapsible({ title, defaultOpen = false, children, badge, className }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={`rounded-md border border-zinc-200 bg-white ${className ?? ""}`}>
      <button
        type="button"
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-medium text-zinc-900 hover:bg-zinc-50 transition-colors"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className={`transition-transform duration-200 ${open ? "rotate-90" : ""}`} aria-hidden="true">
          ▸
        </span>
        <span className="flex-1">{title}</span>
        {badge && (
          <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-500">{badge}</span>
        )}
      </button>
      {open && <div className="border-t border-zinc-100 px-4 pb-4 pt-3">{children}</div>}
    </div>
  );
}