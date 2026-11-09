// @ts-check
import { defineConfig } from 'astro/config';

// https://docs.astro.build/en/reference/configuration-reference/
export default defineConfig({
  // Public address of the site. Repo `gramsc/gramsc.github.io` → served at the root.
  site: 'https://gramsc.github.io',

  // Only needed if the repo is NOT named gramsc.github.io, e.g. repo
  // `gramsc/backpropaganda` → https://gramsc.github.io/backpropaganda/
  // Then uncomment this line (value = '/' + repo name):
  // base: '/backpropaganda',
});
