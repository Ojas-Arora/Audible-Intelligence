// Maps model class labels to user-friendly names and categories
export const FRIENDLY_LABELS: { [key: string]: string } = {
  'airport_image_logmel': 'Airport',
  'bus_image_logmel': 'Bus',
  'metro_image_logmel': 'Metro',
  'park_image_logmel': 'Park',
  'shopping_mall_image_logmel': 'Shopping Mall',
  'unknown': 'Unknown',
};

export const EVENT_CATEGORIES: { [key: string]: string } = {
  'airport_image_logmel': 'transport',
  'bus_image_logmel': 'transport',
  'metro_image_logmel': 'transport',
  'park_image_logmel': 'nature',
  'shopping_mall_image_logmel': 'public',
};

export function getFriendlyLabel(type: string): string {
  return FRIENDLY_LABELS[type] || type.replace(/_/g, ' ').replace(/logmel/i, '').trim() || 'Unknown';
}

export function getCategory(type: string): string {
  return EVENT_CATEGORIES[type] || 'unknown';
}
