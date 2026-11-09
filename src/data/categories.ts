/*
 * The blog's categories, in the order they appear on the home page.
 *
 * Placeholders for now. To rename a category, change `name` / `description`.
 * To add one, add an entry here; a post opts in with `category: <slug>`.
 * If you change a `slug`, update the `category:` line of its posts too
 * (the build will fail with a clear error if a post uses an unknown slug).
 */
export const categories = [
  {
    slug: 'lorem',
    name: 'Lorem ipsum',
    description: 'Lorem ipsum dolor sit amet',
  },
  {
    slug: 'dolor',
    name: 'Dolor sit amet',
    description: 'Consectetur adipiscing elit',
  },
  {
    slug: 'consectetur',
    name: 'Consectetur adipiscing',
    description: 'Sed do eiusmod tempor incididunt',
  },
] as const;

export type Category = (typeof categories)[number];
export type CategorySlug = Category['slug'];

export const categorySlugs = categories.map((c) => c.slug) as [CategorySlug, ...CategorySlug[]];

export function getCategory(slug: string): Category | undefined {
  return categories.find((c) => c.slug === slug);
}

/** "01", "02", ... — the number shown before each category name. */
export function categoryNumber(slug: string): string {
  const index = categories.findIndex((c) => c.slug === slug);
  return String(index + 1).padStart(2, '0');
}
