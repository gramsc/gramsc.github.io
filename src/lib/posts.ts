import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

/** All published posts, newest first. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

/** Published posts in one category, newest first. */
export async function getPostsInCategory(slug: string): Promise<Post[]> {
  return (await getPosts()).filter((p) => p.data.category === slug);
}

/** 2026-11-09 style date, as on the reference layout. */
export function formatDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}
