<script setup lang="ts">
import { ref, watch } from 'vue'

// Define the structure of individual result data
interface web_data {
  score: number;
  url: string;
  text: string;
  title: string;
}

// Define props
defineProps<{ msg: string }>()

// Reactive variables
const query = ref('')
const resultsBM25 = ref<web_data[]>([]) // Left side: BM25 + PageRank
const resultsTFIDF = ref<web_data[]>([]) // Right side: TF-IDF + PageRank
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
      <div class="results-header">
        <span>{{ resultsBM25.length }} results in {{ timeTaken }}s</span>
        <h2>BM25 + PageRank</h2>
      </div>
      <ul class="scrollable">
        <li v-for="result in resultsBM25" :key="result.url">
          <a target="_blank" :href="result.url" class="title">{{ result.title }}</a>
          <p v-html="highlightText(result.text, query)"></p>
        </li>
      </ul>
    </div>

    <!-- Right Side: TF-IDF + PageRank -->
    <div class="results-column">
      <div class="results-header">
        <span>{{ resultsTFIDF.length }} results in {{ timeTaken }}s</span>
        <h2>TF-IDF + PageRank</h2>
      </div>
      <ul class="scrollable">
        <li v-for="result in resultsTFIDF" :key="result.url">
          <a target="_blank" :href="result.url" class="title">{{ result.title }}</a>
          <p v-html="highlightText(result.text, query)"></p>
        </li>
      </ul>
    </div>
  </div>

  <p v-else>No results found</p>

  <p class="credits">Created by Narongchai Rongthong (652115013) for IR assignment 7</p>
</template>

<style scoped>
input {
  width: 70%;
}

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
  text-align: left;
  background: #f9f9f9;
  display: flex;
  flex-direction: column;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.results-header span {
  font-size: 0.9rem;
  color: #444;
}

h2 {
  color: #444;
  margin: 0;
  font-size: 1.1rem;
}

ul {
  list-style: none;
  padding: 0;
  margin: 0;
  flex-grow: 1;
  overflow-y: auto;
}

.scrollable {
  max-height: 400px; /* Adjust this as needed */
}

li {
  margin-bottom: 10px;
}

h3 {
  margin: 0;
  font-size: 1rem;
  color: black;
}

p {
  color: rgb(151, 120, 120);
  font-size: 0.9rem;
  text-align: left;
}

.results-column p {
  color: black;
  font-size: 0.9rem;
  text-align: left;
}

/* Apply the .title color */
.title {
  color: rgb(55, 82, 75);
  font-size: 1rem;
  text-decoration: none;
  cursor: pointer;
}

.title:hover {
  color: #0099cc; /* Change color on hover */
}
</style>
