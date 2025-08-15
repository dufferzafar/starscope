<script setup lang="ts">
import RepoCard from './RepoCard.vue'

export type RepoItem = {
  repo_id: number
  full_name: string
  html_url: string
  description?: string | null
  language?: string | null
  stars?: number | null
  snippet?: string | null
  score?: number | null
}

const props = defineProps<{ items: RepoItem[]; emptyMessage?: string }>()
</script>

<template>
  <div>
    <div v-if="items.length === 0" class="text-sm text-gray-500 dark:text-gray-400 py-8 text-center">
      {{ emptyMessage ?? 'No results yet. Try a query above.' }}
    </div>
    <div v-else class="grid grid-cols-1 gap-3">
      <RepoCard
        v-for="it in items"
        :key="it.repo_id"
        :full-name="it.full_name"
        :html-url="it.html_url"
        :description="it.description"
        :language="it.language"
        :stars="it.stars"
        :snippet="it.snippet"
        :score="it.score"
      />
    </div>
  </div>
</template> 