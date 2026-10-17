// Google Analytics 4, enabled only when the build sets VITE_GA_ID (e.g. in
// web/.env.production). Events carry counts and timings only, never the text
// people type, which stays on their device.

type Params = Record<string, string | number | boolean | undefined>;

const GA_ID = (import.meta.env.VITE_GA_ID as string | undefined)?.trim();

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

const SITE = (import.meta.env.VITE_SITE_URL as string | undefined)?.trim();

export function initAnalytics() {
  if (!GA_ID || !/^G-[A-Z0-9]+$/i.test(GA_ID)) return;
  // only the published site reports; local previews, tests and forks stay silent
  if (SITE && !location.href.toLowerCase().startsWith(SITE.toLowerCase())) return;
  window.dataLayer = window.dataLayer || [];
  // gtag expects the arguments object itself on the data layer
  window.gtag = function gtag() { window.dataLayer!.push(arguments); };
  window.gtag('js', new Date());
  // never report query strings (shared links carry the text someone typed)
  window.gtag('config', GA_ID, { page_location: location.origin + location.pathname });
  const s = document.createElement('script');
  s.async = true;
  s.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(GA_ID)}`;
  document.head.appendChild(s);
  const note = document.getElementById('analytics-note');
  if (note) note.hidden = false;
}

export function track(event: string, params: Params = {}) {
  try {
    window.gtag?.('event', event, params);
  } catch {
    /* analytics must never break the app */
  }
}
