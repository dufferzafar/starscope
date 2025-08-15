<script setup lang="ts">
import { ref } from 'vue'
import SearchBar from './components/SearchBar.vue'
import ResultsList, { type RepoItem } from './components/ResultsList.vue'
import EmptyState from './components/EmptyState.vue'
import ErrorState from './components/ErrorState.vue'
import { useDuckDB } from './composables/useDuckDB'
import { useSearch } from './composables/useSearch'

const query = ref('')
const { initializing, lastError } = useDuckDB()
const { loading, error, search } = useSearch()

const results = ref<RepoItem[]>([])

async function onSearch(q: string) {
  query.value = q
  if (!q) {
    results.value = []
    return
  }
  const rows = await search(q, 20)
  results.value = rows as any
}
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
    <header class="border-b border-gray-200 dark:border-gray-800 bg-white/60 dark:bg-gray-900/50 backdrop-blur">
      <div class="mx-auto max-w-5xl px-4 py-4 flex items-center justify-between">
        <h1 class="text-lg font-semibold">Starscope</h1>
        <a href="https://github.com/dufferzafar" target="_blank" class="text-sm text-blue-700 dark:text-blue-400">GitHub</a>
      </div>
    </header>

    <main class="mx-auto max-w-5xl px-4 py-6 space-y-4">
      <div class="max-w-2xl mx-auto pt-16 pb-8">
        <SearchBar v-model="query" :disabled="loading || initializing" @submit="onSearch" />
        <div class="mt-2 text-center text-xs text-gray-500" v-if="initializing">Loading database…</div>
        <div class="mt-2 text-center text-xs text-red-600" v-if="lastError">{{ lastError }}</div>
      </div>

      <div v-if="error">
        <ErrorState :message="error" />
      </div>
      <div v-else class="max-w-3xl mx-auto">
        <ResultsList :items="results" :empty-message="query ? 'No matches found' : 'Type a query to begin'" />
        <EmptyState v-if="!query && results.length === 0" />
      </div>
    </main>
  </div>
</template>

<style scoped>
</style>
