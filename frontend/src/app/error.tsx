"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Application error:", error);
  }, [error]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        padding: "2rem",
        textAlign: "center",
      }}
    >
      <h2 style={{ fontSize: "1.5rem", fontWeight: 600, marginBottom: "1rem" }}>
        Une erreur est survenue
      </h2>
      <p style={{ color: "#71717a", marginBottom: "1.5rem" }}>
        L&apos;application a rencontré une erreur inattendue. Veuillez réessayer.
      </p>
      <button
        onClick={reset}
        style={{
          padding: "0.5rem 1.5rem",
          backgroundColor: "#2563eb",
          color: "white",
          borderRadius: "0.375rem",
          border: "none",
          cursor: "pointer",
          fontSize: "0.875rem",
          fontWeight: 500,
        }}
      >
        Réessayer
      </button>
    </div>
  );
}