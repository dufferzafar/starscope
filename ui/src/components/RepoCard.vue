<script setup lang="ts">
defineProps<{
  fullName: string
  htmlUrl: string
  description?: string | null
  language?: string | null
  stars?: number | null
  starred_at?: string | null
  snippet?: string | null
  score?: number | null
}>()
</script>

<template>
  <article class="rounded-lg border border-gray-200 dark:border-gray-800 bg-white/60 dark:bg-gray-900/50 p-4 hover:shadow-sm transition">
    <div class="flex items-right justify-between gap-2">
      <a :href="htmlUrl" target="_blank" rel="noopener" class="font-semibold text-blue-700 dark:text-blue-400 hover:underline">{{ fullName }}</a>
      <div class="flex items-center gap-2">
        <span v-if="stars !== null && stars !== undefined" class="inline-flex items-center gap-1 rounded-full bg-gray-100 dark:bg-gray-800 px-2 py-0.5">★ {{ stars }}</span>
        <!-- <span v-if="starred_at" class="inline-flex items-center gap-1 rounded-full bg-gray-100 dark:bg-gray-800 px-2 py-0.5">{{ new Date(starred_at).toISOString().slice(0, 10) }}</span> -->
      </div>
    </div>
    <p v-if="description" class="mt-1 text-sm text-gray-700 dark:text-gray-300">
      <span v-if="language" class="inline-flex items-center gap-1 rounded-full bg-gray-100 dark:bg-gray-800 px-2 py-0.5">{{ language }}</span>
      {{ description }}
    </p>
  </article>
</template> 