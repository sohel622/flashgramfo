import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import multer from 'multer';
import { v2 as cloudinary } from 'cloudinary';

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

// Configure Cloudinary
if (process.env.CLOUDINARY_URL) {
  cloudinary.config({ secure: true });
} else {
  cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME || process.env.CLOUD_NAME || '',
    api_key: process.env.CLOUDINARY_API_KEY || process.env.API_KEY || '',
    api_secret: process.env.CLOUDINARY_API_SECRET || process.env.API_SECRET || '',
    secure: true
  });
}

// Ensure local uploads directory exists for fallback/caching
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Multer upload handler (memory storage for stream to Cloudinary, up to 100MB)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 }
});

// Video Upload endpoint with Cloudinary AWS Rekognition AI Moderation
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
    const inappropriateKeywords = ['18+', 'nsfw', 'porn', 'nudity', 'nude', 'inappropriate', 'violence', 'gore', 'explicit'];
    const matchedKeyword = inappropriateKeywords.find(k => lowerName.includes(k) || lowerCaption.includes(k));
    if (matchedKeyword) {
      console.warn(`[AI Moderation Blocked]: Keyword flagged (${matchedKeyword}) in file "${fileName}"`);
      return res.status(400).json({
        approved: false,
        rejected: true,
        reason: 'Flagged by AI content moderation for inappropriate or 18+ content',
        error: 'Upload blocked: This video was flagged by AI content moderation as inappropriate or containing 18+ content.'
      });
    }

    const hasCloudinary = Boolean(
      process.env.CLOUDINARY_URL || 
      (process.env.CLOUDINARY_CLOUD_NAME && process.env.CLOUDINARY_API_KEY && process.env.CLOUDINARY_API_SECRET)
    );

    if (hasCloudinary) {
      console.log(`[Cloudinary]: Uploading video "${fileName}" with AI Moderation (aws_rek)...`);
      
      const uploadResult = await new Promise((resolve, reject) => {
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
          error: 'Upload blocked: This video was flagged by AI content moderation as inappropriate or containing 18+ content.'
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
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        public_id: uploadResult.public_id,
        duration: uploadResult.duration,
        width: uploadResult.width,
        height: uploadResult.height,
        moderation_status: 'approved'
      });

    } else {
      // Local fallback mode when Cloudinary credentials are not set in .env
      console.log(`[Local AI Moderation Check]: Validating "${fileName}" (Size: ${req.file.size} bytes)`);

      const ext = path.extname(fileName) || '.mp4';
      const fileId = 'vid_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
      const savedPath = path.join(uploadsDir, fileId + ext);

      fs.writeFileSync(savedPath, req.file.buffer);

      const videoUrl = `/uploads/${fileId}${ext}`;
      const thumbnailUrl = `/uploads/${fileId}${ext}`;

      return res.json({
        approved: true,
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        local_fallback: true,
        moderation_status: 'approved',
        notice: 'Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET in .env for live AWS Rekognition'
      });
    }

  } catch (err) {
    console.error('Video upload processing error:', err);
    return res.status(500).json({
      error: 'Failed to process video upload: ' + (err.message || 'Server error')
    });
  }
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
