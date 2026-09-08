/**
 * Tiger's Day – Offline Service Worker (PWA)
 * Enables complete offline play with WebAssembly and local assets.
 */

const CACHE_NAME = 'tigersday-v2.7';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './style.css',
  './script.js',
  './manifest.json',
  './robots.txt',
  './sitemap.xml',
  './og-preview.svg',
  './js/state.js',
  './js/ui/themes.js',
  './js/engine.js',
  './js/scenarios.js',
  './js/mcts.js',
  './js/replay.js',
  './js/lore.js',
  './js/tutorial.js',
  './js/multiplayer.js',
  './js/analytics.js',
  './js/sound.js',
  './opening_book.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('📦 PWA: Pre-caching core game assets');
      return cache.addAll(ASSETS_TO_CACHE).catch((err) => {
        console.warn('PWA: Non-fatal caching warning:', err);
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Skip API calls (handled dynamically)
  if (url.pathname.startsWith('/api/')) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        if (
          !networkResponse ||
          networkResponse.status !== 200 ||
          networkResponse.type !== 'basic'
        ) {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });

        return networkResponse;
      }).catch(() => {
        // Offline fallback
        if (event.request.destination === 'document') {
          return caches.match('./index.html');
        }
      });
    })
  );
});
