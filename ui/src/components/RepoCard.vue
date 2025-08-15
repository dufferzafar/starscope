<script setup lang="ts">
const props = defineProps<{
  fullName: string
  htmlUrl: string
  description?: string | null
  language?: string | null
  stars?: number | null
  snippet?: string | null
  score?: number | null
}>()
</script>

<template>
  <article class="rounded-lg border border-gray-200 dark:border-gray-800 bg-white/60 dark:bg-gray-900/50 p-4 hover:shadow-sm transition">
    <div class="flex items-center justify-between gap-2">
      <a :href="htmlUrl" target="_blank" rel="noopener" class="font-semibold text-blue-700 dark:text-blue-400 hover:underline">{{ fullName }}</a>
      <div v-if="score !== null && score !== undefined" class="text-xs text-gray-500">{{ score.toFixed(3) }}</div>
    </div>
    <p v-if="description" class="mt-1 text-sm text-gray-700 dark:text-gray-300">{{ description }}</p>

    <div class="mt-2 flex flex-wrap items-center gap-3 text-xs text-gray-600 dark:text-gray-400">
      <span v-if="language" class="inline-flex items-center gap-1 rounded-full bg-gray-100 dark:bg-gray-800 px-2 py-0.5">{{ language }}</span>
      <span v-if="stars !== null && stars !== undefined" class="inline-flex items-center gap-1 rounded-full bg-gray-100 dark:bg-gray-800 px-2 py-0.5">★ {{ stars }}</span>
      <span v-if="snippet" class="line-clamp-2 text-gray-500 dark:text-gray-400">{{ snippet }}</span>
    </div>
  </article>
</template> 