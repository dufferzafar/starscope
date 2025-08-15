import { ref } from 'vue'
import { useDuckDB } from './useDuckDB'
import { useEmbedder } from './useEmbedder'

export type Repo = {
  repo_id: number
  full_name: string
  html_url: string
  description?: string | null
  language?: string | null
  stars?: number | null
}

export function useSearch() {
  const { query: sqlQuery } = useDuckDB()
  const { embed } = useEmbedder()
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function ensureSchema(): Promise<{ dim: number }> {
    const arrTable = await sqlQuery("SELECT table_name FROM information_schema.tables WHERE table_name = 'chunk_vectors_arr'")
    if (arrTable.length > 0) {
      const sample = await sqlQuery('SELECT embedding FROM chunk_vectors_arr LIMIT 1')
      if (sample.length > 0) {
        const arr = sample[0].embedding as Float32Array | number[]
        const dim = Array.isArray(arr) ? arr.length : (arr as any).length
        return { dim }
      }
    }
    const s = await sqlQuery('SELECT embedding FROM chunk_vectors LIMIT 1')
    if (s.length > 0) {
      const arr = s[0].embedding as number[]
      return { dim: Array.isArray(arr) ? arr.length : 384 }
    }
    return { dim: 384 }
  }

  function cosine(a: Float32Array, b: Float32Array): number {
    let dot = 0, na = 0, nb = 0
    for (let i = 0; i < a.length; i++) {
      const x = a[i]; const y = b[i]
      dot += x * y; na += x * x; nb += y * y
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb) || 1
    return dot / denom
  }

  async function embedQuery(text: string, dim: number): Promise<Float32Array> {
    const vec = await embed(text)
    if (vec.length === dim) return vec
    const out = new Float32Array(dim)
    out.set(vec.subarray(0, Math.min(vec.length, dim)))
    return out
  }

  async function search(text: string, limit = 20) {
    loading.value = true
    error.value = null
    try {
      const { dim } = await ensureSchema()
      const qvec = await embedQuery(text, dim)

      const hasArr = await sqlQuery("SELECT table_name FROM information_schema.tables WHERE table_name = 'chunk_vectors_arr'")
      if (hasArr.length > 0) {
        const arrLiteral = `[${Array.from(qvec).map((x) => x.toFixed(6)).join(', ')}]::FLOAT[${dim}]`
        const rows = await sqlQuery(
          `WITH q AS (SELECT ${arrLiteral} AS q)
           SELECT c.repo_id, MIN(array_cosine_distance(c.embedding, q.q)) AS dist
           FROM chunk_vectors_arr c, q
           GROUP BY c.repo_id
           ORDER BY dist ASC
           LIMIT ${limit}`
        )
        const ids = rows.map((r: any) => r.repo_id).join(',') || 'NULL'
        const meta = await sqlQuery(`SELECT r.repo_id, r.full_name, r.description, r.language, r.stars, 'https://github.com/' || r.full_name AS html_url FROM repos r WHERE r.repo_id IN (${ids})`)
        const scored = rows.map((r: any) => {
          const m = meta.find((x: any) => x.repo_id === r.repo_id)
          return { ...m, score: 1 - (r.dist ?? 1) }
        })
        scored.sort((a: any, b: any) => (b.score ?? 0) - (a.score ?? 0))
        return scored
      }

      let repoVecs = await sqlQuery(`SELECT cv.repo_id, cv.embedding FROM chunk_vectors cv GROUP BY cv.repo_id HAVING MIN(cv.chunk_id)`)
      const items: Array<{ repo_id: number; score: number }> = []
      for (const row of repoVecs) {
        const emb = row.embedding as number[]
        if (!emb || emb.length !== dim) continue
        const v = Float32Array.from(emb)
        const s = cosine(qvec, v)
        items.push({ repo_id: row.repo_id, score: s })
      }
      items.sort((a, b) => b.score - a.score)
      const top = items.slice(0, limit)
      const ids = top.map((r) => r.repo_id).join(',') || 'NULL'
      const meta = await sqlQuery(`SELECT r.repo_id, r.full_name, r.description, r.language, r.stars, 'https://github.com/' || r.full_name AS html_url FROM repos r WHERE r.repo_id IN (${ids})`)
      return top.map((t) => ({ ...meta.find((m: any) => m.repo_id === t.repo_id), score: t.score }))
    } catch (e: any) {
      error.value = e?.message ?? String(e)
      return []
    } finally {
      loading.value = false
    }
  }

  return { loading, error, search }
} 