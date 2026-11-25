// Supabase Edge Function: cleanup-expired-stories
// POST /functions/v1/cleanup-expired-stories
//
// Deletes expired stories COMPLETELY (story life = 18 hours, set
// by the database trigger in supabase-setup.sql section 14):
//   1. the image file in Storage bucket 'story' FIRST (through
//      the Storage API, using stories.image_path);
//   2. only after the file is confirmed gone: the story's likes,
//      comments, views and the story row itself.
// A failed file delete keeps the row (it is already hidden from
// everyone by RLS), records the error on the row
// (cleanup_attempts / cleanup_last_error / cleanup_last_attempt_at)
// and is retried on later runs with a short back-off — it is
// never given up, so no image file is ever left without its row.
// "File not found" counts as already deleted.
// Then an orphan sweep removes files in 'story' that are older
// than 18 hours and that no story row refers to.
//
// Active stories are never touched: every query and every delete
// is limited to expires_at <= now, and a file that another active
// story still uses is kept.
//
// Called every minute by pg_cron + pg_net (supabase-setup.sql
// section 14.7). Safe to run twice at the same time: deleting a
// file or row that is already gone is a no-op.
//
// Environment (Supabase secrets):
//   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY — provided by Supabase.
//   STORY_CLEANUP_SECRET — a random string; the same value is kept
//     in Supabase Vault as 'story_cleanup_secret' and sent by the
//     cron job as the x-cleanup-secret header. Requests without it
//     are rejected (401), so nobody else can trigger the cleanup.

export const STORY_BUCKET = 'story';
export const STORY_TTL_HOURS = 18;
const MAX_BACKOFF_MINUTES = 60;

// Constant-time string comparison (no early exit on a mismatch).
export function timingSafeEqual(a: string, b: string): boolean {
  const enc = new TextEncoder();
  const x = enc.encode(String(a || ''));
  const y = enc.encode(String(b || ''));
  const len = Math.max(x.length, y.length);
  let diff = x.length ^ y.length;
  for (let i = 0; i < len; i++) diff |= (x[i] || 0) ^ (y[i] || 0);
  return diff === 0 && x.length > 0;
}

// Storage path of a story's image. Prefers image_path; falls back
// to the public URL. Only a file inside the owner's own folder
// counts (same rule as the SQL function story_image_path_from_url).
export function storyFilePath(row: any): string | null {
  const owner = String((row && row.user_id) || '').toLowerCase();
  let p = row && row.image_path ? String(row.image_path) : '';
  if (!p && row && row.image_url) {
    const m = String(row.image_url).match(/\/object\/public\/story\/([^?#]+)/);
    p = m ? m[1] : '';
  }
  if (!p || p.includes('..')) return null;
  if (!owner || p.split('/')[0].toLowerCase() !== owner) return null;
  return p;
}

// Retry back-off after a failed attempt: 1, 2, 4, 8 … up to 60 min.
export function retryDelayMs(attempts: number): number {
  const n = Math.max(0, Number(attempts) || 0);
  if (n === 0) return 0;
  return Math.min(Math.pow(2, n - 1), MAX_BACKOFF_MINUTES) * 60 * 1000;
}

function errText(e: any): string {
  if (!e) return 'unknown error';
  return String(e.message || e.error || e.statusCode || e).slice(0, 500);
}

function isNotFound(e: any): boolean {
  const s = (String((e && (e.message || e.error)) || '') + ' ' + String((e && (e.statusCode || e.status)) || '')).toLowerCase();
  return /not.?found|no such|404/.test(s);
}

export async function runCleanup(sb: any, opts: any = {}) {
  const now: Date = opts.now instanceof Date ? opts.now : new Date();
  const nowMs = now.getTime();
  const nowIso = now.toISOString();
  const batch = opts.batch || 200;
  const orphanLimit = opts.orphanLimit || 500;
  const bucket = sb.storage.from(STORY_BUCKET);
  const summary: any = {
    expired_found: 0, files_deleted: 0, files_already_gone: 0, files_kept_in_use: 0,
    stories_deleted: 0, failed: 0, waiting_retry: 0,
    orphans_found: 0, orphans_deleted: 0, errors: [] as string[]
  };

  // Expired stories only. Never-tried rows first, then the oldest attempt.
  const q = await sb.from('stories')
    .select('id,user_id,image_url,image_path,expires_at,cleanup_attempts,cleanup_last_attempt_at')
    .lte('expires_at', nowIso)
    .order('cleanup_last_attempt_at', { ascending: true, nullsFirst: true })
    .order('expires_at', { ascending: true })
    .limit(batch);
  if (q.error) throw new Error('select expired stories: ' + errText(q.error));
  const rows: any[] = Array.isArray(q.data) ? q.data : [];

  for (const row of rows) {
    // Defensive re-check: an active story is never touched.
    const exp = row && row.expires_at ? Date.parse(row.expires_at) : NaN;
    if (isNaN(exp) || exp > nowMs) continue;
    summary.expired_found++;

    const attempts = Number(row.cleanup_attempts) || 0;
    const last = row.cleanup_last_attempt_at ? Date.parse(row.cleanup_last_attempt_at) : NaN;
    if (attempts > 0 && !isNaN(last) && nowMs - last < retryDelayMs(attempts)) {
      summary.waiting_retry++;
      continue;
    }

    let failure: string | null = null;
    const path = storyFilePath(row);
    if (path) {
      // Same photo still used by another story that is not expired?
      const inUse = await sb.from('stories').select('id')
        .eq('image_path', path).gt('expires_at', nowIso).neq('id', row.id).limit(1);
      if (inUse.error) {
        failure = 'check file in use: ' + errText(inUse.error);
      } else if (Array.isArray(inUse.data) && inUse.data.length) {
        summary.files_kept_in_use++;
      } else {
        const rm = await bucket.remove([path]);
        if (rm.error && !isNotFound(rm.error)) {
          failure = 'storage remove: ' + errText(rm.error);
        } else {
          // Confirm the file is really gone before the row goes.
          const slash = path.lastIndexOf('/');
          const folder = slash >= 0 ? path.slice(0, slash) : '';
          const name = slash >= 0 ? path.slice(slash + 1) : path;
          const ls = await bucket.list(folder, { search: name, limit: 100 });
          if (ls.error) failure = 'storage verify: ' + errText(ls.error);
          else if ((ls.data || []).some((f: any) => f && f.name === name)) failure = 'file still present after remove';
          else if (Array.isArray(rm.data) && rm.data.length) summary.files_deleted++;
          else summary.files_already_gone++;
        }
      }
    }

    if (!failure) {
      // Related rows first (works even without ON DELETE CASCADE).
      for (const t of ['story_likes', 'story_comments', 'story_views']) {
        const d = await sb.from(t).delete().eq('story_id', row.id);
        if (d.error) { failure = 'delete ' + t + ': ' + errText(d.error); break; }
      }
    }
    if (!failure) {
      const d = await sb.from('stories').delete().eq('id', row.id).lte('expires_at', nowIso);
      if (d.error) failure = 'delete story: ' + errText(d.error);
    }

    if (failure) {
      summary.failed++;
      summary.errors.push(String(row.id) + ': ' + failure);
      await sb.from('stories').update({
        cleanup_attempts: attempts + 1,
        cleanup_last_error: failure.slice(0, 500),
        cleanup_last_attempt_at: nowIso
      }).eq('id', row.id);
    } else {
      summary.stories_deleted++;
    }
  }

  // Orphan sweep: files no story row refers to, older than 18h.
  const orph = await sb.rpc('story_orphan_files', { p_min_age_hours: STORY_TTL_HOURS, p_limit: orphanLimit });
  if (orph.error) {
    summary.errors.push('orphan list: ' + errText(orph.error));
  } else {
    const names: string[] = (Array.isArray(orph.data) ? orph.data : [])
      .map((r: any) => (typeof r === 'string' ? r : (r && (r.story_orphan_files || r.name))))
      .filter((n: any) => typeof n === 'string' && n && !n.includes('..'));
    summary.orphans_found = names.length;
    for (let i = 0; i < names.length; i += 100) {
      const chunk = names.slice(i, i + 100);
      const rm = await bucket.remove(chunk);
      if (rm.error && !isNotFound(rm.error)) summary.errors.push('orphan remove: ' + errText(rm.error));
      else summary.orphans_deleted += Array.isArray(rm.data) ? rm.data.length : 0;
    }
  }
  return summary;
}

function json(status: number, body: any) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

export async function handleRequest(req: Request, env: any, makeClient: (url: string, key: string) => Promise<any>) {
  if (req.method !== 'POST') return json(405, { error: 'POST only' });
  const secret = env && env.STORY_CLEANUP_SECRET;
  if (!secret) return json(500, { error: 'STORY_CLEANUP_SECRET is not set' });
  if (!timingSafeEqual(req.headers.get('x-cleanup-secret') || '', secret)) return json(401, { error: 'unauthorized' });
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_ROLE_KEY) return json(500, { error: 'Supabase env missing' });
  try {
    const sb = await makeClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY);
    const summary = await runCleanup(sb);
    return json(200, { ok: true, ...summary });
  } catch (e) {
    return json(500, { ok: false, error: errText(e) });
  }
}

const DenoRt = (globalThis as any).Deno;
if (DenoRt && typeof DenoRt.serve === 'function') {
  DenoRt.serve((req: Request) => handleRequest(req, {
    STORY_CLEANUP_SECRET: DenoRt.env.get('STORY_CLEANUP_SECRET'),
    SUPABASE_URL: DenoRt.env.get('SUPABASE_URL'),
    SUPABASE_SERVICE_ROLE_KEY: DenoRt.env.get('SUPABASE_SERVICE_ROLE_KEY')
  }, async (url: string, key: string) => {
    const { createClient } = await import('https://esm.sh/@supabase/supabase-js@2');
    return createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });
  }));
}
