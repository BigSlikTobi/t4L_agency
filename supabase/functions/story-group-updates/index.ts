import { createClient } from "npm:@supabase/supabase-js@2"

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Content-Type": "application/json",
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: corsHeaders })
}

function resolveMemberIdentifier(row: Record<string, unknown>): string | null {
  for (const key of ["story_id", "url", "cite_url", "member_id", "id"]) {
    const value = row[key]
    if (typeof value === "string" && value.trim().length > 0) {
      return value
    }
  }
  if (typeof row.id === "number") {
    return String(row.id)
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

  const authorization = request.headers.get("Authorization")
  const apikey = request.headers.get("apikey")
  if (!authorization?.startsWith("Bearer ") || !apikey) {
    return jsonResponse({ error: "Unauthorized." }, 401)
  }

  const requestToken = authorization.slice("Bearer ".length).trim()
  if (!requestToken || requestToken !== apikey) {
    return jsonResponse({ error: "Unauthorized." }, 401)
  }

  const expectedToken = Deno.env.get("SUPABASE_FUNCTION_AUTH_TOKEN")
  if (expectedToken && requestToken !== expectedToken) {
    return jsonResponse({ error: "Unauthorized." }, 401)
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL")
  if (!supabaseUrl) {
    return jsonResponse({ error: "Missing SUPABASE_URL." }, 500)
  }

  let payload: { group_id?: unknown; lookback_minutes?: unknown }
  try {
    payload = await request.json()
  } catch {
    return jsonResponse({ error: "Request body must be valid JSON." }, 400)
  }

  if (typeof payload.group_id !== "string" || payload.group_id.trim().length === 0) {
    return jsonResponse({ error: "group_id is required." }, 400)
  }
  const lookbackMinutes = Number(payload.lookback_minutes ?? 60)
  if (!Number.isInteger(lookbackMinutes) || lookbackMinutes <= 0) {
    return jsonResponse({ error: "lookback_minutes must be a positive integer." }, 400)
  }

  const cutoff = new Date(Date.now() - lookbackMinutes * 60 * 1000).toISOString()
  const client = createClient(supabaseUrl, requestToken, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  })

  try {
    const { data, error } = await client
      .schema("vector_embeddings")
      .from("story_group_members")
      .select("*")
      .eq("group_id", payload.group_id.trim())
      .gte("added_at", cutoff)
      .order("added_at", { ascending: false })

    if (error) {
      throw error
    }

    const updates = (data ?? [])
      .map((row) => {
        const record = row as Record<string, unknown>
        const memberIdentifier = resolveMemberIdentifier(record)
        const addedAt = typeof record.added_at === "string" ? record.added_at : null
        if (!memberIdentifier || !addedAt) {
          return null
        }
        return {
          member_identifier: memberIdentifier,
          added_at: addedAt,
        }
      })
      .filter((row): row is { member_identifier: string; added_at: string } => row !== null)

    if (updates.length > 0) {
      return jsonResponse({
        group_id: payload.group_id.trim(),
        lookback_minutes: lookbackMinutes,
        updates,
      })
    }

    const { data: fallbackData, error: fallbackError } = await client
      .from("url_content_lookup")
      .select("url, cite_url, updated_at")
      .eq("group_id", payload.group_id.trim())
      .gte("updated_at", cutoff)
      .order("updated_at", { ascending: false })

    if (fallbackError) {
      throw fallbackError
    }

    const fallbackUpdates = (fallbackData ?? [])
      .map((row) => {
        const record = row as Record<string, unknown>
        const memberIdentifier =
          (typeof record.url === "string" && record.url.trim().length > 0 ? record.url : null) ??
          (typeof record.cite_url === "string" && record.cite_url.trim().length > 0
            ? record.cite_url
            : null)
        const addedAt = typeof record.updated_at === "string" ? record.updated_at : null
        if (!memberIdentifier || !addedAt) {
          return null
        }
        return {
          member_identifier: memberIdentifier,
          added_at: addedAt,
        }
      })
      .filter((row): row is { member_identifier: string; added_at: string } => row !== null)

    return jsonResponse({
      group_id: payload.group_id.trim(),
      lookback_minutes: lookbackMinutes,
      updates: fallbackUpdates,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return jsonResponse({ error: message }, 500)
  }
})
