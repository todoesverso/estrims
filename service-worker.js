const CACHE_NAME = 'estrims-cache-v1';
const urlsToCache = [
  '/',
  '/index.html',
  'https://cdn.jsdelivr.net/npm/bulma@1.0.1/css/bulma.min.css',
];

const addResourcesToCache = async (resources) => {
  const cache = await caches.open(CACHE_NAME);
  await cache.addAll(resources);
};

self.addEventListener('install', event => {
  console.log(`Installing estrims to ${CACHE_NAME}...`);

  event.waitUntil(
    addResourcesToCache(urlsToCache)
  );
});

self.addEventListener('activate', event => {
  console.log('Activate event')
});

const putInCache = async (request, response) => {
  const cache = await caches.open(CACHE_NAME);

  if (request.method === 'POST') {
    console.log('Cannot cache POST requests')
    return
  }

  await cache.put(request, response);
};

const cacheFirst = async (request) => {
  const responseFromCache = await caches.match(request);
  if (responseFromCache) {
    return responseFromCache;
  }
  const responseFromNetwork = await fetch(request);
  // We need to clone the response because the response stream can only be read once
  putInCache(request, responseFromNetwork.clone())
  return responseFromNetwork
};

self.addEventListener("fetch", (event) => {
  event.respondWith(cacheFirst(event.request));
});
