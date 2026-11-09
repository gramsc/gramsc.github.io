/*
 * Defines the "posts" collection: every Markdown file in src/content/posts/
 * is a post, and its frontmatter must match the schema below.
 * A missing or misspelled field makes `npm run dev` / `npm run build`
 * stop with an error pointing at the faulty file.
 */
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { categorySlugs } from './data/categories';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    category: z.enum(categorySlugs),
    summary: z.string(),
    // Set `draft: true` to hide a post from the site while you write it.
    draft: z.boolean().default(false),
  }),
});

export const collections = { posts };
