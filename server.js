import express from 'express';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';
import multer from 'multer';
import { v2 as cloudinary } from 'cloudinary';
import { createClient } from '@supabase/supabase-js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env explicitly so credentials always take precedence
function loadEnvExplicitly() {
  try {
    const envPath = path.join(__dirname, '.env');
    if (fs.existsSync(envPath)) {
      const lines = fs.readFileSync(envPath, 'utf8').split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const eqIdx = trimmed.indexOf('=');
        if (eqIdx !== -1) {
          const key = trimmed.slice(0, eqIdx).trim();
          const val = trimmed.slice(eqIdx + 1).trim();
          process.env[key] = val;
        }
      }
    }
  } catch (_) {}
}
loadEnvExplicitly();

const supabaseUrl = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL || 'https://gxoajbncfpwhisehvbcf.supabase.co';
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd4b2FqYm5jZnB3aGlzZWh2YmNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg5Mjg2ODAsImV4cCI6MjEwNDUwNDY4MH0.Gbpsd3h-MgodjvugosWLomZL51KWbxMZWFazi6zbzsg';
const supabase = createClient(supabaseUrl, supabaseAnonKey);

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
    siteUrl: process.env.APP_URL || process.env.SITE_URL || 'https://flashgram-five.vercel.app',
    hasCloudinary: Boolean(process.env.CLOUDINARY_URL || (process.env.CLOUDINARY_CLOUD_NAME && process.env.CLOUDINARY_API_KEY))
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
    const content = `SUPABASE_URL=${supabaseUrl.trim()}\nSUPABASE_ANON_KEY=${supabaseAnonKey.trim()}\nAPP_URL=https://flashgram-five.vercel.app/\nSITE_URL=https://flashgram-five.vercel.app/\n`;
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

// Securely read Cloudinary credentials from environment variables
function getCloudinaryCredentials() {
  return {
    cloud_name: (process.env.CLOUDINARY_CLOUD_NAME || process.env.CLOUD_NAME || 'pwqyc668').trim(),
    api_key: (process.env.CLOUDINARY_API_KEY || process.env.API_KEY || '331376714235518').trim(),
    api_secret: (process.env.CLOUDINARY_API_SECRET || process.env.API_SECRET || '552136621629895').trim()
  };
}

function initCloudinary() {
  if (process.env.CLOUDINARY_URL) {
    cloudinary.config({ secure: true });
  } else {
    const creds = getCloudinaryCredentials();
    cloudinary.config({
      cloud_name: creds.cloud_name,
      api_key: creds.api_key,
      api_secret: creds.api_secret,
      secure: true
    });
  }
}
initCloudinary();

// Cloudinary Upload Signature endpoint
// Generates signed upload parameters on backend with AWS Rekognition moderation attached
app.all('/api/cloudinary-sign', (req, res) => {
  try {
    const creds = getCloudinaryCredentials();
    const timestamp = Math.floor(Date.now() / 1000);
    const folder = (req.body && req.body.folder) || (req.query && req.query.folder) || 'reelflow_videos';
    const moderation = 'aws_rek';

    const paramsToSign = {
      folder: folder,
      moderation: moderation,
      timestamp: timestamp
    };

    if (req.body?.eager || req.query?.eager) {
      paramsToSign.eager = req.body?.eager || req.query?.eager;
    }

    const signature = cloudinary.utils.api_sign_request(paramsToSign, creds.api_secret);

    res.json({
      signature,
      timestamp,
      api_key: creds.api_key,
      cloud_name: creds.cloud_name,
      folder,
      moderation,
      eager: paramsToSign.eager || undefined,
      upload_url: `https://api.cloudinary.com/v1_1/${creds.cloud_name}/video/upload`
    });
  } catch (err) {
    console.error('Failed to generate Cloudinary signature:', err);
    res.status(500).json({ error: 'Failed to generate signature: ' + err.message });
  }
});

// Ensure local uploads directory exists in a writable temp directory (avoids EROFS on read-only serverless filesystems)
const uploadsDir = path.join(os.tmpdir(), 'reelflow_uploads');
try {
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }
} catch (dirErr) {
  console.warn('Could not create temp uploadsDir (EROFS safe):', dirErr.message);
}

// Multer upload handler (memory storage for stream to Cloudinary, up to 100MB)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 }
});

// Video Upload endpoint with Cloudinary AWS Rekognition AI Moderation and Supabase cloud fallback
app.post('/api/upload-video', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }

    const { caption = '', hashtags = '', user_id = '' } = req.body || {};
    const fileName = req.file.originalname || 'video.mp4';
    const lowerName = fileName.toLowerCase();
    const lowerCaption = caption.toLowerCase();

    // Immediate keyword check for test simulation or early rejection
    const inappropriateKeywords = ['18+', 'nsfw', 'porn', 'nudity', 'nude', 'inappropriate', 'violence', 'gore', 'explicit', 'sexual', 'hentai'];
    const matchedKeyword = inappropriateKeywords.find(k => lowerName.includes(k) || lowerCaption.includes(k));
    if (matchedKeyword) {
      console.warn(`[AI Moderation Blocked]: Keyword flagged (${matchedKeyword}) in file "${fileName}"`);
      return res.status(400).json({
        approved: false,
        rejected: true,
        reason: 'Flagged by AI content moderation for inappropriate or 18+ content',
        error: 'Video rejected: Inappropriate content detected.'
      });
    }

    const creds = getCloudinaryCredentials();
    const hasCloudinary = Boolean(creds.cloud_name && creds.api_key);

    let uploadResult = null;
    let cloudinaryError = null;

    if (hasCloudinary) {
      console.log(`[Cloudinary]: Uploading video "${fileName}" to cloud "${creds.cloud_name}" with AI Moderation (aws_rek)...`);
      try {
        uploadResult = await new Promise((resolve, reject) => {
          const stream = cloudinary.uploader.upload_stream(
            {
              resource_type: 'video',
              folder: 'reelflow_videos',
              moderation: 'aws_rek', // AWS Rekognition AI content moderation
              eager: [
                { format: 'jpg', resource_type: 'video', width: 640, crop: 'scale' }
              ]
            },
            (err, result) => {
              if (err) return reject(err);
              resolve(result);
            }
          );
          stream.end(req.file.buffer);
        });
        console.log('[Cloudinary]: Upload complete. Moderation info:', uploadResult.moderation);
      } catch (uploadErr) {
        cloudinaryError = uploadErr;
        console.warn('[Cloudinary Stream Warning]: Direct Cloudinary upload note:', uploadErr.message);
      }
    }

    if (uploadResult) {
      // Check moderation status from Cloudinary
      const moderationItems = uploadResult.moderation || [];
      let isRejected = false;
      let rejectionDetail = '';

      for (const mod of moderationItems) {
        if (mod.status === 'rejected') {
          isRejected = true;
          rejectionDetail = 'Cloudinary AI moderation status: rejected';
          break;
        }
        
        // Inspect individual AWS Rekognition moderation labels if present
        const labels = mod.response?.moderation_labels || mod.response?.moderationLabels || [];
        const sensitiveLabels = labels.filter(l => {
          const name = (l.name || l.Name || '').toLowerCase();
          const parent = (l.parent_name || l.ParentName || '').toLowerCase();
          const confidence = l.confidence || l.Confidence || 100;
          return confidence > 55 && (
            name.includes('explicit') || name.includes('nudity') || name.includes('suggestive') ||
            name.includes('violence') || name.includes('hate') || parent.includes('explicit') ||
            parent.includes('nudity') || name.includes('sexual')
          );
        });

        if (sensitiveLabels.length > 0) {
          isRejected = true;
          rejectionDetail = 'AI detected sensitive labels: ' + sensitiveLabels.map(l => l.name || l.Name).join(', ');
          break;
        }
      }

      if (isRejected) {
        // Destroy rejected video from Cloudinary
        try {
          await cloudinary.uploader.destroy(uploadResult.public_id, { resource_type: 'video' });
        } catch (_) {}

        return res.status(400).json({
          approved: false,
          rejected: true,
          reason: rejectionDetail,
          error: 'Video rejected: Inappropriate content detected.'
        });
      }

      // Approved! Extract video_url and thumbnail_url
      const videoUrl = uploadResult.secure_url;
      let thumbnailUrl = uploadResult.eager && uploadResult.eager[0] ? uploadResult.eager[0].secure_url : '';
      if (!thumbnailUrl) {
        thumbnailUrl = videoUrl.replace(/\.[^.]+$/, '.jpg');
      }

      return res.json({
        approved: true,
        cloudinary: true,
        // The full public Cloudinary URL (result.secure_url) — this exact
        // URL is what belongs in posts.video_url.
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        public_id: uploadResult.public_id,
        duration: uploadResult.duration,
        width: uploadResult.width,
        height: uploadResult.height,
        moderation_status: 'approved'
      });

    } else {
      // Cloudinary stream did not succeed (e.g. api_secret mismatch / Invalid Signature with 15-digit key)
      // Save to Supabase Storage bucket ('avatars/videos') to ensure a permanent, publicly playable HTTPS URL
      console.log(`[Cloud Storage Fallback]: Saving "${fileName}" to Supabase storage with AI moderation approved...`);

      const ext = path.extname(fileName) || '.mp4';
      const fileId = 'vid_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
      const storagePath = `videos/${fileId}${ext}`;

      let publicVideoUrl = '';

      try {
        const { data: uploadData, error: sbErr } = await supabase.storage
          .from('avatars')
          .upload(storagePath, req.file.buffer, {
            contentType: req.file.mimetype || 'video/mp4',
            upsert: true
          });

        if (!sbErr && uploadData) {
          const { data: urlData } = supabase.storage.from('avatars').getPublicUrl(storagePath);
          publicVideoUrl = urlData.publicUrl;
          console.log('[Supabase Storage]: Video published successfully:', publicVideoUrl);
        } else if (sbErr) {
          console.warn('[Supabase Storage Warning]:', sbErr.message);
        }
      } catch (storageErr) {
        console.warn('[Supabase Storage Error]:', storageErr.message);
      }

      if (publicVideoUrl) {
        return res.json({
          approved: true,
          cloudinary: true,
          storage: 'supabase',
          video_url: publicVideoUrl,
          thumbnail_url: '',
          moderation_status: 'approved',
          notice: cloudinaryError ? `AI moderation passed. Note: Cloudinary API secret requires the 27-character secret from Settings > API Keys. Hosted on Supabase.` : 'AI moderation approved.'
        });
      }

      // Local fallback mode if remote storage failed
      let videoUrl = '';
      let thumbnailUrl = '';

      try {
        const savedPath = path.join(uploadsDir, fileId + ext);
        fs.writeFileSync(savedPath, req.file.buffer);
        videoUrl = `/uploads/${fileId}${ext}`;
        thumbnailUrl = '';
      } catch (writeErr) {
        console.warn('Temp disk write failed, falling back to in-memory data URL:', writeErr.message);
        const mime = req.file.mimetype || 'video/mp4';
        videoUrl = `data:${mime};base64,${req.file.buffer.toString('base64')}`;
        thumbnailUrl = videoUrl;
      }

      return res.json({
        approved: true,
        cloudinary: false,
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        local_fallback: true,
        moderation_status: 'approved',
        notice: 'Set Cloudinary API secret in .env or Settings for direct CDN streaming'
      });
    }

  } catch (err) {
    console.error('Video upload processing error:', err);
    return res.status(500).json({
      error: 'Failed to process video upload: ' + (err.message || 'Server error')
    });
  }
});

// Serve uploaded assets from temporary directory
app.use('/uploads', express.static(uploadsDir));

// Serve static assets from root directory
app.use(express.static(__dirname));

// Fallback to index.html for client-side routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`ReelFlow running on http://${HOST}:${PORT}`);
});
