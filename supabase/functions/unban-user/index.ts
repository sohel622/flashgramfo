// Supabase Edge Function: unban-user
// POST /functions/v1/unban-user
// Body: { targetUserId, banId? }
// The caller must be an authorized admin. The function verifies the
// caller server-side (JWT app_metadata, or the trigger-protected
// profiles.role), sets is_active = false on the ban record(s) in
// public.user_bans (history is kept — the rows stay), and lifts the
// Supabase Auth ban (banned_until = null) via the Auth Admin API.
// It answers ok:true ONLY after BOTH succeed. The service-role key
// lives only in this function's environment — it never reaches the
// frontend.
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

    const supabaseAdmin = createClient(supabaseUrl, serviceRoleKey, {
      auth: { autoRefreshToken: false, persistSession: false },
    });

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
    const banId = body.banId ? String(body.banId) : null;

    if (!UUID_RE.test(targetUserId)) {
      return json(
        { ok: false, error: 'Invalid user ID: not a Supabase Auth UID.' },
        400,
      );
    }

    const nowIso = new Date().toISOString();

    // 1) Deactivate the ban record(s). History is kept — the rows stay
    // with is_active = false.
    let q = supabaseAdmin
      .from('user_bans')
      .update({ is_active: false, updated_at: nowIso })
      .eq('user_id', targetUserId)
      .eq('is_active', true);
    if (banId) q = q.eq('id', banId);
    const { error: updateErr } = await q;
    if (updateErr) {
      return json(
        {
          ok: false,
          error: 'Database update failed: ' + updateErr.message,
        },
        400,
      );
    }

    // 2) Lift the Supabase Auth ban.
    const { error: authErr } =
      await supabaseAdmin.auth.admin.updateUserById(targetUserId, {
        banned_until: null,
      });
    if (authErr) {
      return json(
        {
          ok: false,
          error: 'Auth unban failed: ' + authErr.message,
        },
        400,
      );
    }

    return json({ ok: true });
  } catch (e) {
    return json(
      { ok: false, error: e instanceof Error ? e.message : 'Unban failed.' },
      500,
    );
  }
});
