import { createClient } from "npm:@supabase/supabase-js@2"

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Content-Type": "application/json",
}

type LookupRow = {
  url: string
  cite_url: string | null
  header: string | null
  content: string | null
  description: string | null
  author: string | null
  category: string | null
  quotes: unknown
  embeddings_id: number | null
  group_id: string | null
  created_at: string | null
  updated_at: string | null
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: corsHeaders })
}

function normalizeUrl(value: string): string {
  const trimmed = value.trim()
  if (trimmed.endsWith("/")) {
    return trimmed.slice(0, -1)
  }
  return trimmed
}

function normalizeQuotes(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
  }

  if (typeof value !== "string") {
    return []
  }

  const trimmed = value.trim()
  if (!trimmed || trimmed === "[]") {
    return []
  }

  try {
    const parsed = JSON.parse(trimmed)
    if (Array.isArray(parsed)) {
      return parsed.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
    }
  } catch {
    // Fall back to returning the raw value as a single quote string.
  }

  return [trimmed]
}

function normalizeArticle(row: LookupRow) {
  return {
    url: row.url,
    cite_url: row.cite_url,
    header: row.header,
    content: row.content ?? "",
    description: row.description,
    author: row.author,
    category: row.category,
    quotes: normalizeQuotes(row.quotes),
    embeddings_id: row.embeddings_id,
    group_id: row.group_id,
    created_at: row.created_at,
    updated_at: row.updated_at,
  }
}

async function findArticleByUrl(
  supabaseUrl: string,
  apiKey: string,
  url: string,
): Promise<LookupRow | null> {
  const client = createClient(supabaseUrl, apiKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  })

  const candidates = [...new Set([url, normalizeUrl(url)])].filter(Boolean)
  for (const candidate of candidates) {
    const urlMatch = await client
      .from("url_content_lookup")
      .select(
        "url, cite_url, header, content, description, author, category, quotes, embeddings_id, group_id, created_at, updated_at",
      )
      .eq("url", candidate)
      .limit(1)
      .maybeSingle()
    if (urlMatch.error) {
      throw urlMatch.error
    }
    if (urlMatch.data) {
      return urlMatch.data as LookupRow
    }

    const citeUrlMatch = await client
      .from("url_content_lookup")
      .select(
        "url, cite_url, header, content, description, author, category, quotes, embeddings_id, group_id, created_at, updated_at",
      )
      .eq("cite_url", candidate)
      .limit(1)
      .maybeSingle()
    if (citeUrlMatch.error) {
      throw citeUrlMatch.error
    }
    if (citeUrlMatch.data) {
      return citeUrlMatch.data as LookupRow
    }
  }

  return null
}

Deno.serve(async (request) => {
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  if (request.method !== "POST") {
    return jsonResponse({ error: "Method not allowed." }, 405)
  }

  const expectedToken = Deno.env.get("SUPABASE_FUNCTION_AUTH_TOKEN")
  const authorization = request.headers.get("Authorization")
  const apikey = request.headers.get("apikey")
  if (!authorization?.startsWith("Bearer ") || !apikey) {
    return jsonResponse({ error: "Unauthorized." }, 401)
  }

  const requestToken = authorization.slice("Bearer ".length).trim()
  if (!requestToken || requestToken !== apikey) {
    return jsonResponse({ error: "Unauthorized." }, 401)
  }

  if (expectedToken && requestToken !== expectedToken) {
    return jsonResponse({ error: "Unauthorized." }, 401)
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL")
  if (!supabaseUrl) {
    return jsonResponse({ error: "Missing SUPABASE_URL." }, 500)
  }

  let payload: { url?: unknown }
  try {
    payload = await request.json()
  } catch {
    return jsonResponse({ error: "Request body must be valid JSON." }, 400)
  }

  if (typeof payload.url !== "string" || payload.url.trim().length === 0) {
    return jsonResponse({ error: "url is required." }, 400)
  }

  const requestedUrl = payload.url.trim()

  try {
    const article = await findArticleByUrl(supabaseUrl, requestToken, requestedUrl)
    if (!article) {
      return jsonResponse(
        {
          requested_url: requestedUrl,
          found: false,
          article: null,
        },
        404,
      )
    }

    return jsonResponse({
      requested_url: requestedUrl,
      found: true,
      article: normalizeArticle(article),
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return jsonResponse({ error: message }, 500)
  }
})
