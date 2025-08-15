<script setup lang="ts">
import { ref, watch } from 'vue'

const props = defineProps<{
  modelValue?: string
  placeholder?: string
  disabled?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'submit', value: string): void
}>()

const query = ref(props.modelValue ?? '')

watch(
  () => props.modelValue,
  (v) => {
    if (v !== undefined && v !== query.value) query.value = v
  }
)

function onSubmit() {
  if (props.disabled) return
  emit('submit', query.value.trim())
}
</script>

<template>
  <div class="flex gap-2 w-full">
    <input
      :value="query"
      @input="(e: any) => { query = e.target.value; emit('update:modelValue', query) }"
      @keydown.enter.prevent="onSubmit"
      type="text"
      class="flex-1 px-3 py-2 rounded-md border border-gray-300 dark:border-gray-700 bg-white/90 dark:bg-gray-900/80 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
      :placeholder="placeholder ?? 'Search your stars by topic, task, or stack…'"
      :disabled="disabled"
    />
    <button
      type="button"
      class="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium"
      :disabled="disabled"
      @click="onSubmit"
    >
      Search
    </button>
  </div>
</template> 