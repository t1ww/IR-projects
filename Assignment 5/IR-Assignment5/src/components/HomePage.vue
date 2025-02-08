<script setup lang="ts">
import { ref, watch } from 'vue'
interface web_data {
    score: number
    url: string
    text: string
    title: string
}
interface response {
  elapse: number
  results: web_data[]
}
// Define props
defineProps<{ msg: string }>()

// Reactive variables
const query = ref('')
const resultsBM25 = ref<response[]>([]) // Left side: BM25 + PageRank
const resultsTFIDF = ref<response[]>([]) // Right side: TF-IDF + PageRank
const loading = ref(false)
const error = ref('')
const timeTaken = ref(0)

// Function to fetch API data for both search methods
const fetchResults = async () => {
  if (!query.value.trim()) return

  loading.value = true
  error.value = ''

  try {
    const [bm25Response, tfidfResponse] = await Promise.all([
      fetch(`http://localhost:5000/search_es_pr?query=${query.value}`).then(res => res.json()),
      fetch(`http://localhost:5000/search_manual_pr?query=${query.value}`).then(res => res.json())
    ])

    console.log("BM25 Response:", bm25Response)  // Debugging
    console.log("TF-IDF Response:", tfidfResponse)  // Debugging

    // Ensure we're accessing the "results" array properly
    resultsBM25.value = bm25Response.results ?? []
    resultsTFIDF.value = tfidfResponse.results ?? []
    
    // Capture the time taken
    timeTaken.value = bm25Response.elapse ?? tfidfResponse.elapse ?? 0
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    loading.value = false
  }
}


// Auto-fetch when `query` changes
watch(query, fetchResults)

// Function to highlight query terms in search results
const highlightText = (text: string, query: string) => {
  if (!text || !query) return text // Early exit if text or query is invalid
  const regex = new RegExp(`(${query})`, 'gi') // Make case insensitive search
  return text.replace(regex, '<b>$1</b>') // Highlight the query text
}
</script>

<template>
  <h1>{{ msg }}</h1>

  <div class="card">
    <p>Start your search here</p>
    <input v-model="query" placeholder="Type to search...">
  </div>

  <p v-if="loading">Loading...</p>
  <p v-if="error" class="error">{{ error }}</p>
  <p v-if="!loading && (resultsBM25.length || resultsTFIDF.length)">
    Found {{ resultsBM25.length + resultsTFIDF.length }} results in {{ timeTaken }}s
  </p>

  <div v-if="resultsBM25.length || resultsTFIDF.length" class="results-container">
    <!-- Left Side: BM25 + PageRank -->
    <div class="results-column">
      <h2>BM25 + PageRank</h2>
      <ul>
        <li v-for="result in resultsBM25" :key="result.id">
          <h3>{{ result.title }}</h3>
          <p v-html="highlightText(result.snippet)"></p>
        </li>
      </ul>
    </div>

    <!-- Right Side: TF-IDF + PageRank -->
    <div class="results-column">
      <h2>TF-IDF + PageRank</h2>
      <ul>
        <li v-for="result in resultsTFIDF" :key="result.id">
          <h3>{{ result.title }}</h3>
          <p v-html="highlightText(result.snippet)"></p>
        </li>
      </ul>
    </div>
  </div>

  <p v-else>No results found</p>

  <p class="credits">Created by Narongchai Rongthong (652115013) for IR assignment 7</p>
</template>

<style scoped>
.credits { color: #888; }
.error { color: red; }

.results-container {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-top: 20px;
}

.results-column {
  width: 48%;
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 8px;
  background: #f9f9f9;
}

h2 {
  color: #444;
}

ul {
  list-style: none;
  padding: 0;
}

li {
  margin-bottom: 10px;
}

h3 {
  margin: 0;
  font-size: 1.1rem;
}

p {
  color: #666;
  font-size: 0.9rem;
}
</style>
