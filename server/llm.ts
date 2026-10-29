// The ONLY place that talks to an AI provider. Everything else calls generateJson().
// The key arrives per request from the browser. Never store it, log it or return it.
import { GoogleGenAI } from "@google/genai";
import type { Request } from "express";
import type { ZodType } from "zod";
import { ApiError } from "./envelope.js";

export type Provider = "gemini" | "openrouter" | "anthropic";
export type LlmConfig = { provider: Provider; key: string; model: string };

export const DEFAULT_MODELS = { gemini: "gemini-3.5-flash-lite", openrouter: "google/gemini-3.5-flash-lite", anthropic: "claude-sonnet-5-5" } as const;

export function configFromRequest(req: Request): LlmConfig {
  const provider = String(req.header("x-llm-provider") ?? "").toLowerCase();
  const key = String(req.header("x-llm-key") ?? "").trim();
  const model = String(req.header("x-llm-model") ?? "").trim();
  if (!key) throw new ApiError(400, "MISSING_KEY", "Add your key in Settings.");
  if (provider !== "gemini" && provider !== "openrouter" && provider !== "anthropic") throw new ApiError(400, "BAD_PROVIDER", "Choose gemini, openrouter or anthropic in Settings.");
  return { provider, key, model: model || DEFAULT_MODELS[provider] };
}

function stripFences(text: string): string {
  return text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
}

// Turns a provider failure into a clear message that says what to fix. Never includes the key.
function providerError(status: number, raw: string, cfg: LlmConfig): ApiError {
  let reason = raw;
  try { const j = JSON.parse(raw.slice(raw.indexOf("{"))); reason = j?.error?.message ?? j?.message ?? raw; } catch { /* plain text */ }
  reason = String(reason).split(cfg.key).join("***").replace(/\s+/g, " ").trim().slice(0, 180);
  const low = reason.toLowerCase();
  const suggest = DEFAULT_MODELS[cfg.provider];
  if (status === 429 || low.includes("quota") || low.includes("resource_exhausted") || low.includes("rate limit"))
    return new ApiError(429, "RATE_LIMITED", `Too many requests or quota used up for this key. Wait a minute or use another key. (${reason})`);
  if (status === 401 || status === 403)
    return new ApiError(401, "BAD_KEY", `The provider rejected this key. Check you picked the right provider and pasted the whole key. (${reason})`);
  if (status === 404 || low.includes("not found") || low.includes("is not supported") || low.includes("no longer available") || low.includes("not a valid model"))
    return new ApiError(400, "BAD_MODEL", `Model "${cfg.model}" is not available for this key. Set Model to ${suggest} in Settings. (${reason})`);
  if (low.includes("api key") || low.includes("api_key") || low.includes("permission") || low.includes("unauthorized") || low.includes("authentication"))
    return new ApiError(401, "BAD_KEY", `The provider rejected this key. Check you picked the right provider and pasted the whole key. (${reason})`);
  if (status === 402 || low.includes("credit") || low.includes("billing"))
    return new ApiError(402, "NO_CREDIT", `This key has no credit left. (${reason})`);
  if (!status) return new ApiError(502, "PROVIDER_UNREACHABLE", `Could not reach the AI provider. Check your internet connection. (${reason})`);
  return new ApiError(502, "PROVIDER_ERROR", `The AI provider failed (HTTP ${status}). (${reason})`);
}

async function callProvider(prompt: string, cfg: LlmConfig): Promise<string> {
  if (cfg.provider === "gemini") {
    try {
      const ai = new GoogleGenAI({ apiKey: cfg.key });
      const r = await ai.models.generateContent({ model: cfg.model, contents: prompt, config: { responseMimeType: "application/json" } });
      return r.text ?? "";
    } catch (e: any) {
      throw providerError(Number(e?.status ?? e?.code ?? 0), String(e?.message ?? e), cfg);
    }
  }
  if (cfg.provider === "anthropic") {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": cfg.key, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({ model: cfg.model, max_tokens: 8000, messages: [{ role: "user", content: prompt }] }),
    });
    if (!r.ok) throw providerError(r.status === 529 ? 429 : r.status, await r.text(), cfg);
    const j: any = await r.json();
    return (j?.content ?? []).filter((b: any) => b.type === "text").map((b: any) => b.text).join("");
  }
  const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: cfg.model, messages: [{ role: "user", content: prompt }], response_format: { type: "json_object" } }),
  });
  if (!r.ok) throw providerError(r.status, await r.text(), cfg);
  const j: any = await r.json();
  if (j?.error) throw providerError(Number(j.error.code) || 502, JSON.stringify(j), cfg);
  return j?.choices?.[0]?.message?.content ?? "";
}

export async function generateJson<T>(prompt: string, schema: ZodType<T>, cfg: LlmConfig): Promise<T> {
  let last = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    const suffix = attempt === 0 ? "" : `\n\nYour previous answer was invalid: ${last}. Reply with valid JSON only.`;
    const raw = await callProvider(prompt + "\n\nReply with JSON only, no commentary." + suffix, cfg);
    try {
      const parsed = schema.safeParse(JSON.parse(stripFences(raw)));
      if (parsed.success) return parsed.data;
      last = parsed.error.issues.slice(0, 3).map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
    } catch {
      last = "not parseable JSON";
    }
  }
  throw new ApiError(502, "BAD_AI_OUTPUT", "The AI did not return valid JSON. Try again or pick another model.");
}
