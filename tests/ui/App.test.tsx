import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "../../src/client/App.js";
import { DEFAULT_SETTINGS } from "../../src/server/defaultSettings.js";

describe("App", () => {
  it("renders the cockpit shell", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.endsWith("/api/sessions")) return response([]);
        if (url.endsWith("/api/settings")) return response(DEFAULT_SETTINGS);
        return response({});
      })
    );
    render(<App />);
    expect(await screen.findByText("AIOps Cockpit")).toBeInTheDocument();
    expect(screen.getByText("Local control plane")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});

function response(data: unknown) {
  return {
    ok: true,
    json: async () => data
  } as Response;
}
