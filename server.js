import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env if present
try {
  process.loadEnvFile();
} catch (_) {}

const app = express();
const PORT = process.env.DEFAULT_APP_PORT || (process.env.PORT && String(process.env.PORT) !== '8080' ? Number(process.env.PORT) : 3000);
const HOST = '0.0.0.0';

app.use(express.json());

// Serve local Supabase UMD bundle
app.get('/vendor/supabase.js', (req, res) => {
  const localBundle = path.join(__dirname, 'node_modules/@supabase/supabase-js/dist/umd/supabase.js');
  if (fs.existsSync(localBundle)) {
    res.sendFile(localBundle);
  } else {
    res.status(404).send('Supabase bundle not found');
  }
});

// Config endpoint for client
app.get('/api/config', (req, res) => {
  res.json({
    configured: Boolean((process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL) && (process.env.SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_ANON_KEY)),
    supabaseUrl: process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL || '',
    supabaseAnonKey: process.env.SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_ANON_KEY || '',
    siteUrl: process.env.APP_URL || process.env.SITE_URL || 'https://flashgram-five.vercel.app'
  });
});

app.post('/api/config', (req, res) => {
  const { supabaseUrl, supabaseAnonKey } = req.body || {};
  if (!supabaseUrl || !supabaseAnonKey) {
    return res.status(400).json({ error: 'supabaseUrl and supabaseAnonKey are required' });
  }

  process.env.SUPABASE_URL = supabaseUrl.trim();
  process.env.SUPABASE_ANON_KEY = supabaseAnonKey.trim();

  try {
    const envPath = path.join(__dirname, '.env');
    const content = `PORT=${PORT}\nSUPABASE_URL=${supabaseUrl.trim()}\nSUPABASE_ANON_KEY=${supabaseAnonKey.trim()}\n`;
    fs.writeFileSync(envPath, content, 'utf8');
  } catch (err) {
    console.warn('Could not write .env:', err);
  }

  res.json({
    success: true,
    configured: true,
    supabaseUrl: process.env.SUPABASE_URL,
    supabaseAnonKey: process.env.SUPABASE_ANON_KEY
  });
});

// Serve static assets from root directory
app.use(express.static(__dirname));

// Fallback to index.html for client-side routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`ReelFlow running on http://${HOST}:${PORT}`);
});
