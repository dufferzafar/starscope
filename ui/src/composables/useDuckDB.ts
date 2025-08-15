import { ref } from 'vue'

// Lazy singleton for DuckDB connection
let duckdbReady: Promise<any> | null = null
let duckdbConn: any | null = null

function getPublicUrl(path: string): string {
	const base = import.meta.env.BASE_URL || '/'
	return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`
}

export function useDuckDB() {
	const initializing = ref(false)
	const lastError = ref<string | null>(null)

	async function init() {
		if (duckdbConn) return duckdbConn
		if (!duckdbReady) {
			initializing.value = true
			lastError.value = null
			duckdbReady = (async () => {
				const duckdb = await import('@duckdb/duckdb-wasm')
				// Import worker/module URLs locally so they are served from the same origin (Vite dev server or Pages)
				const mainModule = (await import('@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url')).default as string
				const workerUrl = (await import('@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url')).default as string

				const logger = new (duckdb as any).ConsoleLogger()
				const worker = new Worker(workerUrl, { type: 'module' })
				const db = new (duckdb as any).AsyncDuckDB(logger, worker)
				await db.instantiate(mainModule)

				const dbUrl = getPublicUrl('data/starscope.duckdb')
				// Fetch DB as binary and register buffer explicitly
				const res = await fetch(dbUrl)
				if (!res.ok) {
					throw new Error(`Failed to fetch DB: ${res.status} ${res.statusText}`)
				}
				const buf = new Uint8Array(await res.arrayBuffer())
				await db.registerFileBuffer('starscope.duckdb', buf)
				await db.open({ path: 'starscope.duckdb', accessMode: 'readonly' })
				duckdbConn = await db.connect()
				initializing.value = false
				return duckdbConn
			})().catch((e: any) => {
				lastError.value = e?.message ?? String(e)
				initializing.value = false
				throw e
			})
		}
		return duckdbReady
	}

	async function query(sql: string) {
		const conn = await init()
		const result = await conn.query(sql)
		return result.toArray?.() ?? result
	}

	return {
		initializing,
		lastError,
		init,
		query,
	}
} 