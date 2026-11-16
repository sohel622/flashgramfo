// Supabase Edge Function: ban-user
// POST /functions/v1/ban-user
// Body: { targetUserId, banReason, banType: 'temporary' | 'permanent',
//         expiresAt: ISO string | null }
// The caller must be an authorized admin. The function verifies the
// caller server-side (JWT app_metadata, or the trigger-protected
// profiles.role), verifies the target user exists in Supabase Auth,
// deactivates any existing active ban for that user, inserts the ban
// into public.user_bans, and sets the Auth banned_until via the
// Auth Admin API. It answers ok:true ONLY after BOTH the database
// insert AND the Auth ban succeed; if the Auth ban fails, the
// inserted row is rolled back. The service-role key lives only in
// this function's environment — it never reaches the frontend.
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers':
    'authorization, x-client-info, apikey, content-type',
};

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }
  if (req.method !== 'POST') {
    return json({ ok: false, error: 'Method not allowed' }, 405);
  }

  try {
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? '';
    const anonKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '';
    if (!supabaseUrl || !anonKey || !serviceRoleKey) {
      return json(
        { ok: false, error: 'Server is missing Supabase configuration.' },
        500,
      );
    }

    const authHeader = req.headers.get('Authorization') ?? '';
    if (!authHeader) {
      return json({ ok: false, error: 'Missing Authorization header.' }, 401);
    }

    // Caller client (anon key + the caller's JWT) — identifies the admin.
    const supabase = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: authHeader } },
      auth: { autoRefreshToken: false, persistSession: false },
    });
    const {
      data: { user: caller },
      error: callerErr,
    } = await supabase.auth.getUser();
    if (callerErr || !caller) {
      return json({ ok: false, error: 'Invalid session.' }, 401);
    }

    // Admin client (service role) — the only one that can touch Auth
    // admin operations and bypass RLS on user_bans.
    const supabaseAdmin = createClient(supabaseUrl, serviceRoleKey, {
      auth: { autoRefreshToken: false, persistSession: false },
    });

    // Admin check: protected app_metadata first (only settable via the
    // Auth Admin API / Dashboard, never by the user), then the
    // trigger-protected profiles.role column.
    const meta = (caller.app_metadata ?? {}) as Record<string, unknown>;
    let isAdmin = meta.is_admin === true || meta.role === 'admin';
    if (!isAdmin) {
      const { data: prof } = await supabaseAdmin
        .from('profiles')
        .select('role')
        .eq('user_id', caller.id)
        .maybeSingle();
      isAdmin = prof?.role === 'admin';
    }
    if (!isAdmin) {
      return json({ ok: false, error: 'Admin only.' }, 403);
    }

    const body = await req.json().catch(() => ({}));
    const targetUserId = String(body.targetUserId ?? '').trim();
    const banReason = String(body.banReason ?? '').trim();
    const banType = String(body.banType ?? 'temporary').trim();
    const expiresAt = body.expiresAt ? String(body.expiresAt) : null;

    if (!UUID_RE.test(targetUserId)) {
      return json(
        { ok: false, error: 'Invalid user ID: not a Supabase Auth UID.' },
        400,
      );
    }
    if (!banReason) {
      return json({ ok: false, error: 'Ban reason is required.' }, 400);
    }
    if (banType !== 'temporary' && banType !== 'permanent') {
      return json(
        { ok: false, error: "banType must be 'temporary' or 'permanent'." },
        400,
      );
    }
    if (banType === 'temporary' && !expiresAt) {
      return json(
        { ok: false, error: 'A temporary ban needs an expiry date.' },
        400,
      );
    }
    if (banType === 'temporary' && Number.isNaN(Date.parse(expiresAt!))) {
      return json({ ok: false, error: 'Invalid expiry date.' }, 400);
    }

    // Verify the target user actually exists in Supabase Auth.
    const { data: target, error: targetErr } =
      await supabaseAdmin.auth.admin.getUserById(targetUserId);
    if (targetErr || !target.user) {
      return json(
        {
          ok: false,
          error: 'No user with this UID exists in Supabase Auth.',
        },
        404,
      );
    }
    if (target.user.id === caller.id) {
      return json(
        { ok: false, error: 'An admin cannot ban themself.' },
        400,
      );
    }

    const nowIso = new Date().toISOString();
    const bannedUntil =
      banType === 'permanent'
        ? '2999-01-01T00:00:00.000Z'
        : new Date(expiresAt!).toISOString();

    // Deactivate any existing active ban for this user first, so there
    // is exactly one active ban per user (history is kept).
    await supabaseAdmin
      .from('user_bans')
      .update({ is_active: false, updated_at: nowIso })
      .eq('user_id', targetUserId)
      .eq('is_active', true);

    // 1) Insert the ban record.
    const { data: ban, error: insertErr } = await supabaseAdmin
      .from('user_bans')
      .insert({
        user_id: targetUserId,
        ban_reason: banReason,
        ban_type: banType,
        banned_at: nowIso,
        expires_at: banType === 'permanent' ? null : bannedUntil,
        is_active: true,
        created_by: caller.id,
      })
      .select(
        'id,user_id,ban_reason,ban_type,banned_at,expires_at,is_active,created_by,created_at',
      )
      .single();
    if (insertErr || !ban) {
      return json(
        {
          ok: false,
          error: 'Database insert failed: ' + (insertErr?.message ?? 'unknown'),
        },
        400,
      );
    }

    // 2) Apply the Supabase Auth ban (banned_until). Only after BOTH
    // succeed do we report success.
    const { error: authErr } =
      await supabaseAdmin.auth.admin.updateUserById(targetUserId, {
        banned_until: bannedUntil,
      });
    if (authErr) {
      // Roll the database row back so there is no half-done ban.
      await supabaseAdmin.from('user_bans').delete().eq('id', ban.id);
      return json(
        {
          ok: false,
          error: 'Auth ban failed: ' + authErr.message,
        },
        400,
      );
    }

    return json({ ok: true, ban });
  } catch (e) {
    return json(
      { ok: false, error: e instanceof Error ? e.message : 'Ban failed.' },
      500,
    );
  }
});
