/*
 * Internal links must go through `url()` so they keep working if the site
 * is served from a sub-folder (the `base` option in astro.config.mjs),
 * e.g. url('/about/') → '/about/' at the root, '/backpropaganda/about/' in a sub-folder.
 */
const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export function url(path: string): string {
  return base + (path.startsWith('/') ? path : `/${path}`);
}

/** The current path with the base removed, for "which page am I on" checks. */
export function stripBase(pathname: string): string {
  return base && pathname.startsWith(base) ? pathname.slice(base.length) || '/' : pathname;
}
