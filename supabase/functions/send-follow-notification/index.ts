// Supabase Edge Function: send-follow-notification
// POST /functions/v1/send-follow-notification
// Called by a Database Webhook on INSERT into public.followers
// (webhook payload: { type: 'INSERT', table: 'followers',
//   record: { follower_id, following_id, created_at } }).
// It looks up the follower's profile (name, username, avatar) and
// the followed user's FCM device tokens (public.device_tokens),
// then sends "[Name] started following you." to each Android
// device via Firebase Cloud Messaging (HTTP v1 API).
//
// The app never sends a notification itself: the Firebase service
// account that can send messages lives only in this function's
// environment — it never reaches the frontend.
//
// Environment (Supabase secrets):
//   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
//   FCM_SERVICE_ACCOUNT_JSON — the full service-account JSON from
//     Firebase Console > Project settings > Service accounts
//     (client_email, private_key, project_id).
//   FOLLOW_WEBHOOK_SECRET — a random string, also set as the
//     x-webhook-secret header on the Database Webhook. Requests
//     without it are rejected, so nobody else can trigger sends.
//
// Duplicates: the followers unique index blocks a duplicate
// follow row, the webhook only fires on INSERT, and this function
// records one notification_log row per follow — a retried webhook
// (same follower + following) returns deduped:true and sends
// nothing again.
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers':
    'authorization, x-client-info, apikey, content-type, x-webhook-secret',
};

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

function base64urlEncode(bytes: Uint8Array): string {
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(
      null,
      Array.from(bytes.subarray(i, i + CHUNK)),
    );
  }
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function base64DecodeToBytes(b64: string): Uint8Array {
  const bin = atob(b64.replace(/\s/g, ''));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// Short-lived OAuth2 access token for the FCM HTTP v1 API, signed
// from the Firebase service account (RS256 JWT, Google's flow).
async function getFcmAccessToken(
  serviceAccount: { client_email: string; private_key: string; project_id: string },
): Promise<{ accessToken: string; projectId: string }> {
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: 'RS256', typ: 'JWT' };
  const claims = {
    iss: serviceAccount.client_email,
    scope: 'https://www.googleapis.com/auth/firebase.messaging',
    aud: 'https://oauth2.googleapis.com/token',
    exp: now + 3600,
    iat: now,
  };
  const enc = new TextEncoder();
  const unsigned = base64urlEncode(enc.encode(JSON.stringify(header))) +
    '.' +
    base64urlEncode(enc.encode(JSON.stringify(claims)));
  const pem = serviceAccount.private_key
    .replace(/-----BEGIN PRIVATE KEY-----/g, '')
    .replace(/-----END PRIVATE KEY-----/g, '')
    .replace(/\\n/g, '\n')
    .replace(/\s/g, '');
  const key = await crypto.subtle.importKey(
    'pkcs8',
    base64DecodeToBytes(pem),
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const sig = await crypto.subtle.sign(
    'RSASSA-PKCS1-v1_5',
    key,
    enc.encode(unsigned),
  );
  const jwt = unsigned + '.' + base64urlEncode(new Uint8Array(sig));
  const res = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
      assertion: jwt,
    }),
  });
  if (!res.ok) {
    throw new Error('FCM OAuth token exchange failed: ' + res.status);
  }
  const data = await res.json();
  return { accessToken: String(data.access_token), projectId: serviceAccount.project_id };
}

async function sendFcmMessage(
  accessToken: string,
  projectId: string,
  token: string,
  title: string,
  body: string,
  image: string | null,
  data: Record<string, string>,
): Promise<{ ok: boolean; dead: boolean; detail: string }> {
  const message: Record<string, unknown> = {
    token,
    notification: { title, body },
    data,
    android: { priority: 'HIGH', notification: {} },
  };
  if (image) {
    (message.notification as Record<string, unknown>).image = image;
    ((message.android as Record<string, unknown>).notification as Record<string, unknown>).image = image;
  }
  const res = await fetch(
    `https://fcm.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/messages:send`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    },
  );
  const text = await res.text();
  if (res.ok) return { ok: true, dead: false, detail: text };
  // UNREGISTERED / NOT_FOUND: the device uninstalled the app or
  // the token expired — the token row is deleted by the caller.
  const dead = /UNREGISTERED|NOT_FOUND/i.test(text);
  return { ok: false, dead, detail: text.slice(0, 300) };
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
    const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '';
    const fcmJson = Deno.env.get('FCM_SERVICE_ACCOUNT_JSON') ?? '';
    const webhookSecret = Deno.env.get('FOLLOW_WEBHOOK_SECRET') ?? '';
    if (!supabaseUrl || !serviceRoleKey) {
      return json(
        { ok: false, error: 'Server is missing Supabase configuration.' },
        500,
      );
    }
    if (!fcmJson) {
      return json(
        { ok: false, error: 'Server is missing the FCM service account.' },
        500,
      );
    }
    if (!webhookSecret) {
      return json(
        { ok: false, error: 'Server is missing the webhook secret.' },
        500,
      );
    }
    if (req.headers.get('x-webhook-secret') !== webhookSecret) {
      return json({ ok: false, error: 'Unauthorized webhook caller.' }, 401);
    }

    const payload = await req.json().catch(() => ({}));
    if (payload.type && payload.type !== 'INSERT') {
      return json({ ok: true, ignored: true });
    }
    const record = payload.record ?? payload ?? {};
    const followerId = String(record.follower_id ?? '').trim();
    const followingId = String(record.following_id ?? '').trim();
    if (!UUID_RE.test(followerId) || !UUID_RE.test(followingId)) {
      return json({ ok: false, error: 'Invalid follower/following IDs.' }, 400);
    }
    if (followerId === followingId) {
      return json({ ok: true, ignored: true });
    }

    const supabaseAdmin = createClient(supabaseUrl, serviceRoleKey, {
      auth: { autoRefreshToken: false, persistSession: false },
    });

    // Idempotency first: one row per follow. A retried webhook
    // hits the unique key and sends nothing again.
    const dedupeKey = `follow:${followerId}:${followingId}`;
    const { error: logErr } = await supabaseAdmin
      .from('notification_log')
      .insert({ dedupe_key: dedupeKey, kind: 'follow' });
    if (logErr && String(logErr.code) === '23505') {
      return json({ ok: true, deduped: true });
    }
    if (logErr) {
      return json(
        { ok: false, error: 'Could not record the notification: ' + logErr.message },
        500,
      );
    }

    // The follower's name, username and profile picture.
    const { data: follower } = await supabaseAdmin
      .from('profiles')
      .select('display_name, username, avatar_url')
      .eq('user_id', followerId)
      .maybeSingle();
    const name = String(
      (follower && (follower.display_name || follower.username)) || 'Someone',
    );
    const username = String((follower && follower.username) || '');
    const avatarUrl = String((follower && follower.avatar_url) || '') || null;

    // Every Android device the followed user signed in on.
    const { data: tokenRows } = await supabaseAdmin
      .from('device_tokens')
      .select('token')
      .eq('user_id', followingId);
    const tokens = (tokenRows ?? [])
      .map((r) => String(r.token || '').trim())
      .filter(Boolean);
    if (tokens.length === 0) {
      return json({ ok: true, sent: 0, note: 'No device tokens.' });
    }

    let serviceAccount: { client_email: string; private_key: string; project_id: string };
    try {
      serviceAccount = JSON.parse(fcmJson);
    } catch {
      return json({ ok: false, error: 'FCM service account is not valid JSON.' }, 500);
    }
    if (!serviceAccount.client_email || !serviceAccount.private_key || !serviceAccount.project_id) {
      return json({ ok: false, error: 'FCM service account is incomplete.' }, 500);
    }
    const { accessToken, projectId } = await getFcmAccessToken(serviceAccount);

    const title = `${name} started following you.`;
    const body = username ? `@${username}` : '';
    let sent = 0;
    const failures: string[] = [];
    for (const token of tokens) {
      const result = await sendFcmMessage(accessToken, projectId, token, title, body, avatarUrl, {
        type: 'follow',
        follower_id: followerId,
        follower_username: username,
      });
      if (result.ok) {
        sent += 1;
      } else {
        failures.push(result.detail);
        if (result.dead) {
          // Dead token: stop trying it on every future follow.
          await supabaseAdmin.from('device_tokens').delete().eq('token', token);
        }
      }
    }
    return json({ ok: true, sent, tokens: tokens.length, failures: failures.slice(0, 3) });
  } catch (err) {
    console.error('send-follow-notification error:', err);
    return json(
      { ok: false, error: (err as Error).message || 'Unexpected error.' },
      500,
    );
  }
});
