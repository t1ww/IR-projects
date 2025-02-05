<script setup lang="ts">
import { ref, watch } from 'vue'
import type web_data from '@/types'

// Define props
defineProps<{ msg: string }>()

// Reactive variables
const query = ref('')
const results = ref<web_data[]>([])  // Store API results
const loading = ref(false)
const error = ref('')

// Function to fetch API data
const fetchResults = async () => {
  if (!query.value.trim()) return

  loading.value = true
  error.value = ''

  try {
    const response = await fetch(`https://localhost:5000/search?q=${query.value}`)
    if (!response.ok) throw new Error('Failed to fetch data')

    results.value = await response.json()
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    loading.value = false
  }
}

// Auto-fetch when `query` changes
watch(query, fetchResults)
</script>

<template>
  <h1>{{ msg }}</h1>

  <div class="card">
    <p>Start your search here</p>
    <input v-model="query" placeholder="Type to search...">
  </div>

  <p v-if="loading">Loading...</p>
  <p v-if="error" class="error">{{ error }}</p>

  <ul v-if="results.length">
    <li v-for="result in results" :key="result.id">{{ result.title }}</li>
  </ul>

  <p v-else>No results found</p>

  <p class="credits">Created by Narongchai Rongthong (652115013) for IR assignment 7</p>
</template>

<style scoped>
.credits { color: #888; }
.error { color: red; }
</style>
