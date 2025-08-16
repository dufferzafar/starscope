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
  starred_at?: string | null
}

export function useSearch() {
  const { query: sqlQuery } = useDuckDB()
  const { embed } = useEmbedder()
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function ensureSchema(): Promise<{ dim: number }> {
    // Prefer fixed-size array table from VSS build
    const arrTable = await sqlQuery("SELECT table_name FROM information_schema.tables WHERE table_name = 'repo_reps_arr'")
    if (arrTable.length > 0) {
      const sample = await sqlQuery('SELECT embedding FROM repo_reps_arr LIMIT 1')
      if (sample.length > 0) {
        const arr = sample[0].embedding as Float32Array | number[]
        const dim = Array.isArray(arr) ? arr.length : (arr as any).length
        return { dim }
      }
    }
    // Fallback to base table with LIST(FLOAT)
    const s = await sqlQuery('SELECT embedding FROM repo_reps LIMIT 1')
    if (s.length > 0) {
      const arr = s[0].embedding as number[]
      return { dim: Array.isArray(arr) ? arr.length : 384 }
    }
    return { dim: 384 }
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

      // Prefer VSS array table if present
      const hasArr = await sqlQuery("SELECT table_name FROM information_schema.tables WHERE table_name = 'repo_reps_arr'")
      const arrLiteral = `[${Array.from(qvec).map((x) => x.toFixed(6)).join(', ')}]::FLOAT[${dim}]`
      if (hasArr.length > 0) {
        const rows = await sqlQuery(
          `WITH q AS (SELECT ${arrLiteral} AS q)
           SELECT c.repo_id, MIN(array_cosine_distance(c.embedding, q.q)) AS dist
           FROM repo_reps_arr c, q
           GROUP BY c.repo_id
           ORDER BY dist ASC
           LIMIT ${limit}`
        )
        const ids = rows.map((r: any) => r.repo_id).join(',') || 'NULL'
        const meta = await sqlQuery(`SELECT s.repo_id, s.full_name, s.description, s.language, s.stargazers_count AS stars, s.starred_at, 'https://github.com/' || s.full_name AS html_url FROM stars s WHERE s.repo_id IN (${ids})`)
        const scored = rows.map((r: any) => {
          const m = meta.find((x: any) => x.repo_id === r.repo_id)
          return { ...m, score: 1 - (r.dist ?? 1) }
        })
        scored.sort((a: any, b: any) => (b.score ?? 0) - (a.score ?? 0))
        return scored
      }

      // No VSS array table; run full-scan cosine in DuckDB with cast to FLOAT[dim]
      const rows = await sqlQuery(
        `WITH q AS (SELECT ${arrLiteral} AS q)
         SELECT c.repo_id, MIN(array_cosine_distance(CAST(c.embedding AS FLOAT[${dim}]), q.q)) AS dist
         FROM repo_reps c, q
         GROUP BY c.repo_id
         ORDER BY dist ASC
         LIMIT ${limit}`
      )
      const ids = rows.map((r: any) => r.repo_id).join(',') || 'NULL'
      const meta = await sqlQuery(`SELECT s.repo_id, s.full_name, s.description, s.language, s.stargazers_count AS stars, s.starred_at, 'https://github.com/' || s.full_name AS html_url FROM stars s WHERE s.repo_id IN (${ids})`)
      const scored = rows.map((r: any) => {
        const m = meta.find((x: any) => x.repo_id === r.repo_id)
        return { ...m, score: 1 - (r.dist ?? 1) }
      })
      scored.sort((a: any, b: any) => (b.score ?? 0) - (a.score ?? 0))
      return scored
    } catch (e: any) {
      error.value = e?.message ?? String(e)
      return []
    } finally {
      loading.value = false
    }
  }

  return { loading, error, search }
} 