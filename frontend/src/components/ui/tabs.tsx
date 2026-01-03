import * as React from "react";

import { cn } from "@/lib/utils";

type TabsContextValue = {
  value: string;
  onValueChange?: (value: string) => void;
};

const TabsContext = React.createContext<TabsContextValue | null>(null);

const Tabs = ({ value, onValueChange, children }: React.PropsWithChildren<TabsContextValue>) => (
  <TabsContext.Provider value={{ value, onValueChange }}>{children}</TabsContext.Provider>
);

const TabsList = ({ className, children }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("inline-flex items-center gap-2 rounded-full bg-zinc-100 p-1", className)}>
    {children}
  </div>
);

type TabsTriggerProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { value: string };

const TabsTrigger = ({ value, className, children, ...props }: TabsTriggerProps) => {
  const ctx = React.useContext(TabsContext);
  if (!ctx) {
    throw new Error("TabsTrigger must be used within Tabs");
  }
  const active = ctx.value === value;
  return (
    <button
      type="button"
      onClick={() => ctx.onValueChange?.(value)}
      className={cn(
        "rounded-full px-3 py-1.5 text-xs font-medium transition",
        active ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-900",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};

const TabsContent = ({ value, className, children }: { value: string } & React.HTMLAttributes<HTMLDivElement>) => {
  const ctx = React.useContext(TabsContext);
  if (!ctx) {
    throw new Error("TabsContent must be used within Tabs");
  }
  if (ctx.value !== value) {
    return null;
  }
  return <div className={cn("pt-4", className)}>{children}</div>;
};

export { Tabs, TabsContent, TabsList, TabsTrigger };
