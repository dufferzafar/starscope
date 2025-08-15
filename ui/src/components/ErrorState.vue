<script setup lang="ts">
const props = defineProps<{ message?: string }>()
</script>

<template>
  <div class="text-center text-red-600 dark:text-red-400 py-4">
    {{ message ?? 'Something went wrong.' }}
  </div>
</template> 