<script setup lang="ts">
import RepoCard from './RepoCard.vue'
import { computed, ref } from 'vue'

export type RepoItem = {
  repo_id: number
  full_name: string
  html_url: string
  description?: string | null
  language?: string | null
  stars?: number | null
  starred_at?: string | null
  snippet?: string | null
  score?: number | null
}

const props = defineProps<{ items: RepoItem[]; emptyMessage?: string }>()

const sortBy = ref<'score' | 'stars'>('score')

const sortedItems = computed(() => {
  const key = sortBy.value
  const list = [...props.items]
  return list.sort((a, b) => {
    const aVal = (key === 'stars' ? a.stars : a.score) ?? -1
    const bVal = (key === 'stars' ? b.stars : b.score) ?? -1
    return bVal - aVal
  })
})
</script>

<template>
  <div>
    <div v-if="items.length === 0" class="text-sm text-gray-500 dark:text-gray-400 py-8 text-center">
      {{ emptyMessage ?? 'No results yet. Try a query above.' }}
    </div>
    <div v-else>
      <div class="flex justify-end pb-2">
        <label class="sr-only" for="sort">Sort by</label>
        <select id="sort" v-model="sortBy" class="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-sm px-2 py-1">
          <option value="score">Sort by score</option>
          <option value="stars">Sort by stars</option>
        </select>
      </div>
      <div class="grid grid-cols-1 gap-3">
        <RepoCard
          v-for="it in sortedItems"
          :key="it.repo_id"
          :full-name="it.full_name"
          :html-url="it.html_url"
          :description="it.description"
          :language="it.language"
          :stars="it.stars"
          :starred_at="it.starred_at"
          :snippet="it.snippet"
          :score="it.score"
        />
      </div>
    </div>
  </div>
</template> 